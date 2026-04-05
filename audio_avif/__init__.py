import os
import math
import json
import warnings
import numpy as np
import soundfile as sf
import librosa
from PIL import Image
from scipy.ndimage import gaussian_filter1d

try:
    import pillow_avif  # noqa: F401
    HAS_AVIF = True
except ImportError:
    pillow_avif = None
    HAS_AVIF = False

# Constants
TARGET_SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 80
FMAX = 7600
MIN_DB = -11.0
MAX_DB = 4.0
QUALITIES = [70, 80, 85, 90, 95]
DEFAULT_GRIFFIN_LIM_ITERS = 32
DECODER_CHOICES = ("vocoder", "griffin-lim")

def get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_vocoder(device=None):
    from transformers import SpeechT5HifiGan

    if device is None:
        device = get_device()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()
    return vocoder

def wav_to_logmel(wav_path, n_mels=N_MELS):
    """
    Reads wav, returns ((T, n_mels) log10-mel spectrogram, rms).
    rms is calculated on the 16kHz audio used for mel extraction.
    """
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1) # mono
    wav = wav.astype(np.float32)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # Calculate RMS
    rms = np.sqrt(np.mean(wav**2))

    # Compute STFT
    S = np.abs(
        librosa.stft(
            wav,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    ).astype(np.float64, copy=False)

    # Mel basis
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=FMAX,
        htk=False,         # Slaney
        norm="slaney",     # Slaney
    ).astype(np.float64, copy=False)

    mel = np.einsum("mf,ft->mt", mel_basis, S, optimize=True)
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel) # (n_mels, T)

    return logmel.T.astype(np.float32), float(rms) # Return (T, n_mels), rms

def wav_to_mfcc(wav_path, n_mfcc=N_MELS, n_mels=N_MELS):
    """
    Reads wav, returns ((T, n_mfcc) mfcc coefficients, rms).
    """
    logmel, rms = wav_to_logmel(wav_path, n_mels=n_mels)
    # logmel is (T, n_mels). Transpose to (n_mels, T) for librosa
    # librosa expects log-power mel spectrogram in dB when S is provided.
    # wav_to_logmel uses log10 of amplitude-like mel, so convert to power-dB.
    logmel_T = (logmel * 20.0).T
    
    # MFCC (DCT Type-II, Orthogonal normalization)
    mfcc = librosa.feature.mfcc(S=logmel_T, n_mfcc=n_mfcc, dct_type=2, norm='ortho')
    
    return mfcc.T.astype(np.float32), rms

def mfcc_to_logmel(mfcc, n_mels=N_MELS):
    """
    Converts (T, n_mfcc) MFCC back to (T, n_mels) Log-Mel spectrogram.
    """
    mfcc_T = mfcc.T
    mel_T = librosa.feature.inverse.mfcc_to_mel(mfcc_T, n_mels=n_mels, dct_type=2, norm='ortho')
    mel_T = np.maximum(mel_T, 1e-10)
    # mfcc_to_mel returns power; convert to log10 amplitude-like mel used elsewhere.
    logmel_T = 0.5 * np.log10(mel_T)
    return logmel_T.T.astype(np.float32)

def matrix_to_image(data, min_val=None, max_val=None, metadata=None, reshape=False, stretch=1.0, gaussian_blur=None, horizontal_usm=None, shift_key=0):
    """
    Generic function to convert a 2D matrix (T, H) to a PIL Image.
    """
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    
    if metadata is None:
        metadata = {}
        
    # Store essential reconstruction info
    metadata['min_val'] = float(min_val)
    metadata['max_val'] = float(max_val)
    metadata['height'] = data.shape[1]

    # Normalize to 0-1
    # Avoid division by zero
    rng = max_val - min_val
    if rng == 0:
        rng = 1.0
        
    norm = (data - min_val) / rng
    norm = np.clip(norm, 0.0, 1.0)
    
    # Scale to 0-255
    uint8_data = (norm * 255.0).astype(np.uint8)
    
    # Orientation: Data is (T, H). Image expects (H, T).
    # Flipud so index 0 (low freq/coeff) is at bottom.
    img_data = np.flipud(uint8_data.T) # (H, T)
    
    H_img, T_img = img_data.shape

    # Apply shifts sequentially
    s_keys = [shift_key] if isinstance(shift_key, int) else shift_key
    for sk in s_keys:
        if sk == 0:
            continue
        # Estimate noise floor (5th percentile)
        noise_floor = np.percentile(img_data, 5)
        shifted = np.full_like(img_data, int(noise_floor))
        
        if sk > 0:
            # Shift Up
            if sk < H_img:
                shifted[:-sk, :] = img_data[sk:, :]
        else:
            # Shift Down
            neg_sk = -sk
            if neg_sk < H_img:
                shifted[neg_sk:, :] = img_data[:-neg_sk, :]
        img_data = shifted
    
    if gaussian_blur is not None:
        kernel_size, sigma = gaussian_blur
        if sigma > 0:
            truncate = ((kernel_size - 1) / 2.0) / sigma
            img_data = gaussian_filter1d(img_data.astype(np.float32), sigma=sigma, axis=1, truncate=truncate)
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)

    if horizontal_usm is not None:
        kernel_size, sigma, strength = horizontal_usm
        if strength != 0:
            truncate = ((kernel_size - 1) / 2.0) / sigma
            img_data_f = img_data.astype(np.float32)
            blurred = gaussian_filter1d(img_data_f, sigma=sigma, axis=1, truncate=truncate)
            img_data = img_data_f + (img_data_f - blurred) * strength
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)

    original_width = img_data.shape[1]
    
    if stretch != 1.0:
        new_w = int(round(original_width * stretch))
        # Use PIL for resizing
        img_temp = Image.fromarray(img_data)
        img_temp = img_temp.resize((new_w, H_img), resample=Image.Resampling.BILINEAR)
        img_data = np.array(img_temp)
    
    current_width = img_data.shape[1]

    if reshape:
        # Square heuristic
        T = current_width
        # Target roughly square: Side ~ sqrt(H * T)
        # Number of strips k = Side / H = sqrt(T/H)
        val = T / float(H_img)
        k = max(1, int(round(math.sqrt(val))))
        
        # Calculate width per strip
        width_per_strip = math.ceil(T / k)
        
        # Align to 16 for compression block efficiency
        if width_per_strip % 16 != 0:
            width_per_strip = ((width_per_strip // 16) + 1) * 16
            
        total_width_needed = width_per_strip * k
        pad_amount = total_width_needed - T
        
        if pad_amount > 0:
            padding = np.zeros((H_img, pad_amount), dtype=np.uint8)
            img_data = np.hstack([img_data, padding])
            
        # Split and Stack Vertically
        chunks = [img_data[:, i*width_per_strip : (i+1)*width_per_strip] for i in range(k)]
        img_data = np.vstack(chunks) # (H*k, width_per_strip)
        
        metadata['orig_w'] = current_width

    img = Image.fromarray(img_data)

    exif = img.getexif()
    exif[270] = json.dumps(metadata)
    # Caller must save with exif=img.getexif()
    
    return img

def logmel_to_image(logmel, min_val=MIN_DB, max_val=MAX_DB, rms=None, reshape=False, stretch=1.0, gaussian_blur=None, horizontal_usm=None, shift_key=0):
    """
    Wrapper for backward compatibility.
    """
    metadata = {}
    if rms is not None:
        metadata['rms'] = rms
    metadata['type'] = 'mel'
    metadata['mel_bins'] = int(logmel.shape[1])
        
    return matrix_to_image(logmel, min_val, max_val, metadata, reshape, stretch, gaussian_blur, horizontal_usm, shift_key)

def logmel_to_webp_anim(logmel, rms, output_path, quality=80, chunk_width=64, gaussian_blur=None, horizontal_usm=None, shift_key=0):
    """
    Saves logmel as an animated WebP (pseudo-video).
    logmel: (T, n_mels) float
    """
    # 1. Normalize and quantize
    min_val, max_val = MIN_DB, MAX_DB
    norm = (logmel - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0)
    uint8_data = (norm * 255.0).astype(np.uint8)
    
    # Orientation: (n_mels, T)
    img_data = np.flipud(uint8_data.T) # (n_mels, T)
    H, T = img_data.shape

    s_keys = [shift_key] if isinstance(shift_key, int) else shift_key
    for sk in s_keys:
        if sk == 0:
            continue
        noise_floor = np.percentile(img_data, 5)
        shifted = np.full_like(img_data, int(noise_floor))
        if sk > 0:
            if sk < H:
                shifted[:-sk, :] = img_data[sk:, :]
        else:
            neg_sk = -sk
            if neg_sk < H:
                shifted[neg_sk:, :] = img_data[:-neg_sk, :]
        img_data = shifted

    if gaussian_blur is not None:
        kernel_size, sigma = gaussian_blur
        if sigma > 0:
            truncate = ((kernel_size - 1) / 2.0) / sigma
            img_data = gaussian_filter1d(img_data.astype(np.float32), sigma=sigma, axis=1, truncate=truncate)
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)

    if horizontal_usm is not None:
        kernel_size, sigma, strength = horizontal_usm
        if strength != 0:
            truncate = ((kernel_size - 1) / 2.0) / sigma
            img_data_f = img_data.astype(np.float32)
            blurred = gaussian_filter1d(img_data_f, sigma=sigma, axis=1, truncate=truncate)
            img_data = img_data_f + (img_data_f - blurred) * strength
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)

    # 2. Chunking
    frames = []
    
    # Ensure all chunks are same size by padding the last one if needed
    # WebP/Video often prefers even dimensions. chunk_width should be even.
    num_chunks = math.ceil(T / chunk_width)
    
    for i in range(num_chunks):
        start = i * chunk_width
        end = start + chunk_width
        chunk = img_data[:, start:end]
        
        # Pad if last chunk is smaller
        if chunk.shape[1] < chunk_width:
            pad_w = chunk_width - chunk.shape[1]
            padding = np.zeros((H, pad_w), dtype=np.uint8)
            chunk = np.hstack([chunk, padding])
            
        frames.append(Image.fromarray(chunk))
        
    # 3. Metadata
    metadata = {
        'rms': rms,
        'orig_w': T,
        'chunk_w': chunk_width,
        'height': H,
        'mel_bins': H,
        'type': 'mel',
    }
    exif_bytes = None
    
    # Create a dummy image to generate EXIF bytes
    dummy = Image.new('L', (1,1))
    exif = dummy.getexif()
    exif[270] = json.dumps(metadata)
    exif_bytes = exif.tobytes()

    # 4. Save
    # Pillow supports saving animated WebP.
    # We use lossy compression (quality < 100).
    # method=6 is slowest/best compression.
    if frames:
        frames[0].save(
            output_path,
            format='WEBP',
            save_all=True,
            append_images=frames[1:],
            duration=33, # roughly 30fps, doesn't matter for audio reconstruction but needed for players
            loop=0,
            quality=quality,
            method=6,
            exif=exif_bytes
        )

def webp_anim_to_logmel(image_path, min_val=MIN_DB, max_val=MAX_DB):
    """
    Reads animated WebP, reconstructs logmel (T, n_mels).
    """
    img = Image.open(image_path)
    
    # Extract Metadata
    rms = None
    orig_w = None
    
    exif = img.getexif()
    if exif and 270 in exif:
        try:
            meta = json.loads(exif[270])
            rms = meta.get('rms')
            orig_w = meta.get('orig_w')
        except:
            pass
            
    # Iterate frames
    frames_data = []
    try:
        while True:
            # Convert to grayscale and numpy
            frame = img.convert('L')
            frames_data.append(np.array(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
        
    if not frames_data:
        raise ValueError("No frames found in WebP")
        
    # Concatenate horizontally
    # Each frame is (H, chunk_w)
    full_img = np.hstack(frames_data) # (H, total_w)
    
    # Crop to original width
    if orig_w is not None:
        full_img = full_img[:, :orig_w]
        
    # Convert back to logmel
    # Flip back
    full_img = np.flipud(full_img)
    
    # Transpose (H, T) -> (T, H)
    logmel_norm = full_img.astype(np.float32).T / 255.0
    logmel = logmel_norm * (max_val - min_val) + min_val
    
    return logmel, rms

def image_to_matrix(image, default_min=MIN_DB, default_max=MAX_DB):
    """
    Generic function to convert PIL Image to (T, H) data matrix.
    Reads metadata for reconstruction parameters.
    """
    # Parse Metadata
    rms = None
    orig_w = None
    min_val = default_min
    max_val = default_max
    height = None
    data_type = None
    mel_bins = None
    
    exif = image.getexif()
    if exif and 270 in exif:
        desc = exif[270]
        # Try Parsing JSON
        try:
            meta = json.loads(desc)
            if isinstance(meta, dict):
                rms = meta.get('rms')
                orig_w = meta.get('orig_w')
                if 'min_val' in meta:
                    min_val = meta['min_val']
                if 'max_val' in meta:
                    max_val = meta['max_val']
                height = meta.get('height')
                data_type = meta.get('type')
                mel_bins = meta.get('mel_bins')
        except json.JSONDecodeError:
            # Fallback to legacy
            if isinstance(desc, str) and desc.startswith("OriginalRMS:"):
                try:
                    rms = float(desc.split(":")[1])
                except:
                    pass

    image = image.convert('L')
    img_data = np.array(image) # (H_img, W_img)
    H_img, W_img = img_data.shape
    
    # Try to infer height if not in metadata
    if height is None:
        # Assume 80 for legacy files
        height = 80
        
    # Un-reshape if needed
    if orig_w is not None:
        # We know each strip is 'height' pixels high
        if H_img % height == 0:
            k = H_img // height
            # Split vertically
            chunks = np.vsplit(img_data, k)
            # Stack horizontally
            img_data = np.hstack(chunks) # (height, k*W_strip)
            # Crop padding
            img_data = img_data[:, :orig_w]
        else:
            print(f"Warning: Image height {H_img} is not a multiple of {height}, cannot un-reshape correctly. Treating as standard.")

    img_data = img_data.astype(np.float32)
    
    # Flip back (Spectrogram was flipud)
    img_data = np.flipud(img_data)
    
    # Transpose back: (H, T) -> (T, H)
    norm_data = img_data.T / 255.0
    
    # De-normalize
    data = norm_data * (max_val - min_val) + min_val
    
    return data, {
        'rms': rms,
        'min_val': min_val,
        'max_val': max_val,
        'height': height,
        'type': data_type,
        'mel_bins': mel_bins if mel_bins is not None else height,
    }

def image_to_logmel(image, min_val=MIN_DB, max_val=MAX_DB):
    """
    Wrapper for backward compatibility. Returns (logmel, rms).
    """
    data, metadata = image_to_matrix(image, min_val, max_val)
    return data, metadata.get('rms')

def reconstruct_wav_with_vocoder(logmel, vocoder, device):
    """
    logmel: (T, 80)
    """
    import torch

    if logmel.shape[-1] != N_MELS:
        raise ValueError(
            f"SpeechT5 vocoder expects {N_MELS} mel bins, got {logmel.shape[-1]}. "
            "Use --decoder griffin-lim for non-80-bin log-mel."
        )

    spectrogram = torch.tensor(logmel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        waveform = vocoder(spectrogram)
        
    return waveform.squeeze().cpu().numpy()

def reconstruct_wav_with_griffin_lim(logmel, griffin_lim_iters=DEFAULT_GRIFFIN_LIM_ITERS):
    """
    Reconstructs waveform from log-mel without a neural vocoder.
    """
    mel = np.power(10.0, np.asarray(logmel, dtype=np.float64).T).astype(np.float64, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul.*",
            category=RuntimeWarning,
        )
        magnitude = librosa.feature.inverse.mel_to_stft(
            mel,
            sr=TARGET_SR,
            n_fft=N_FFT,
            power=1.0,
            fmin=FMIN,
            fmax=FMAX,
            htk=False,
            norm="slaney",
        )
    if not np.isfinite(magnitude).all():
        raise ValueError("mel_to_stft produced non-finite magnitudes")
    waveform = librosa.griffinlim(
        magnitude,
        n_iter=griffin_lim_iters,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        window="hann",
        center=True,
        pad_mode="reflect",
        random_state=0,
    )
    return np.asarray(waveform, dtype=np.float32)

def reconstruct_wav(
    logmel,
    vocoder=None,
    device=None,
    decoder="vocoder",
    griffin_lim_iters=DEFAULT_GRIFFIN_LIM_ITERS,
):
    """
    Reconstructs waveform from (T, 80) log-mel.
    """
    if decoder == "vocoder":
        if vocoder is None:
            raise ValueError("vocoder decoder requires a loaded vocoder instance")
        return reconstruct_wav_with_vocoder(logmel, vocoder, device)

    if decoder == "griffin-lim":
        return reconstruct_wav_with_griffin_lim(
            logmel,
            griffin_lim_iters=griffin_lim_iters,
        )

    raise ValueError(
        f"Unsupported decoder '{decoder}'. Expected one of: {', '.join(DECODER_CHOICES)}"
    )

def apply_loudness(wav, target_rms):
    """
    Adjusts wav loudness to match target_rms.
    """
    if target_rms is None:
        return wav
        
    current_rms = np.sqrt(np.mean(wav**2))
    if current_rms <= 1e-9:
        return wav
        
    gain = target_rms / current_rms
    return wav * gain
