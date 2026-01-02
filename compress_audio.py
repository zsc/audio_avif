import argparse
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from transformers import SpeechT5HifiGan
from PIL import Image
import pillow_avif  # Ensures plugin is registered

# Constants from mel.md
TARGET_SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 80
FMAX = 7600
MIN_DB = -11.0  # Estimated min for log10(1e-10) is -10. We use -11 to be safe/consistent
MAX_DB = 4.0    # Estimated max. log10(10000) is 4. usually signals are normalized so < 1. 
                # Wait, if signal is [-1, 1], magnitude is <= 1?
                # If power=1.0 (magnitude), max val is sum of window. 
                # Hann window sum is N/2 approx. 1024/2 = 512.
                # log10(512) approx 2.7. 
                # So 4.0 is a safe upper bound.
QUALITIES = [70, 80, 85, 90, 95]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def wav_to_logmel(wav_path):
    """
    Reads wav, returns (T, 80) log10-mel spectrogram.
    Strictly follows mel.md specs.
    """
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1) # mono
    wav = wav.astype(np.float32)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

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
    )

    # Mel basis
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=False,         # Slaney
        norm="slaney",     # Slaney
    )

    mel = mel_basis @ S
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel) # (80, T)

    return logmel.T.astype(np.float32) # Return (T, 80)

def logmel_to_image(logmel, min_val=MIN_DB, max_val=MAX_DB):
    """
    Converts (T, 80) float logmel to PIL Image (Grayscale).
    Time axis maps to Width. Frequency axis maps to Height.
    So Image size will be (T, 80).
    Input shape: (T, 80)
    """
    # Normalize to 0-1
    norm = (logmel - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0)
    
    # Scale to 0-255
    uint8_data = (norm * 255.0).astype(np.uint8)
    
    # Transpose to (80, T) for Image.fromarray which expects (H, W)
    # But wait, usually we want low freq at bottom.
    # index 0 is low freq in librosa.
    # In image, (0,0) is top-left.
    # So index 0 will be at top. 
    # To have low freq at bottom, we should flipud the (80, T) array.
    img_data = np.flipud(uint8_data.T) # (80, T) -> freq is height, time is width
    
    return Image.fromarray(img_data, mode='L')

def image_to_logmel(image, min_val=MIN_DB, max_val=MAX_DB):
    """
    Converts PIL Image to (T, 80) logmel.
    """
    image = image.convert('L')
    img_data = np.array(image).astype(np.float32)
    
    # Flip back (we flipped up-down to put low freq at bottom)
    img_data = np.flipud(img_data)
    
    # Transpose back: (80, T) -> (T, 80)
    logmel_norm = img_data.T / 255.0
    
    # De-normalize
    logmel = logmel_norm * (max_val - min_val) + min_val
    
    return logmel

def reconstruct_wav(logmel, vocoder, device):
    """
    logmel: (T, 80)
    """
    # Prepare input for vocoder
    # Vocoder expects (batch, seq_len, dim)
    spectrogram = torch.tensor(logmel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        waveform = vocoder(spectrogram)
        
    return waveform.squeeze().cpu().numpy()

def generate_html(output_dir, results):
    """
    Generates index.html
    results: list of dicts { 'name': 'filename', 'original': 'path/to/wav', 'variants': { '70': {'avif': '...', 'wav': '...'}, ... } }
    """
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Compression via Mel-Spectrogram Image</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .sample { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
        .controls { margin-bottom: 15px; }
        .comparison { display: flex; gap: 20px; }
        .panel { flex: 1; }
        .spectrogram-container { position: relative; margin-top: 10px; background: #000; overflow-x: auto; }
        .spectrogram-img { display: block; height: 150px; width: 100%; object-fit: cover; image-rendering: pixelated; }
        .cursor { position: absolute; top: 0; bottom: 0; width: 2px; background: red; left: 0; pointer-events: none; }
        audio { width: 100%; margin-top: 5px; }
        h3 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Audio Compression via AVIF Mel-Spectrogram</h1>
        <p>Using <code>microsoft/speecht5_hifigan</code> for vocoding.</p>
        
        <div id="samples">
            <!-- Samples will be injected here -->
        </div>
    </div>

    <script>
        const data = DATA_PLACEHOLDER;

        function renderSamples() {
            const container = document.getElementById('samples');
            data.forEach((sample, index) => {
                const div = document.createElement('div');
                div.className = 'sample';
                
                // Default quality
                const defaultQ = "90"; 
                
                div.innerHTML = `
                    <h2>${sample.name}</h2>
                    <div class="controls">
                        <label>AVIF Quality: 
                            <select id="sel-${index}" onchange="updateSample(${index})">
                                ${Object.keys(sample.variants).map(q => `<option value="${q}" ${q === defaultQ ? 'selected' : ''}>${q}</option>`).join('')}
                            </select>
                        </label>
                    </div>
                    
                    <div class="comparison">
                        <div class="panel">
                            <h3>Original (Resampled 16kHz)</h3>
                            <audio id="audio-orig-${index}" controls src="${sample.original}" ontimeupdate="updateCursor(${index}, 'orig')"></audio>
                            <div class="spectrogram-container" id="spec-container-orig-${index}">
                                <img src="${sample.original_mel}" class="spectrogram-img"> 
                                <div id="cursor-orig-${index}" class="cursor"></div>
                            </div>
                        </div>
                        
                        <div class="panel">
                            <h3>Reconstructed (from AVIF Q<span id="lbl-q-${index}">${defaultQ}</span>)</h3>
                            <audio id="audio-recon-${index}" controls src="${sample.variants[defaultQ].wav}" ontimeupdate="updateCursor(${index}, 'recon')"></audio>
                            <div class="spectrogram-container" id="spec-container-recon-${index}">
                                <img id="img-recon-${index}" src="${sample.variants[defaultQ].avif}" class="spectrogram-img">
                                <div id="cursor-recon-${index}" class="cursor"></div>
                            </div>
                            <div id="info-${index}" style="margin-top: 10px; font-family: monospace; background: #eee; padding: 10px; border-radius: 4px;">
                                <!-- Sizes and Ratio will be injected here -->
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(div);
                
                // Init size and info
                setTimeout(() => updateSample(index), 100);
            });
        }
        
        function updateSample(index) {
            const sel = document.getElementById(`sel-${index}`);
            // Safety check in case element isn't ready
            if (!sel) return;
            
            const q = sel.value;
            const sample = data[index];
            const variant = sample.variants[q];
            
            // Update Right Side
            document.getElementById(`lbl-q-${index}`).innerText = q;
            document.getElementById(`audio-recon-${index}`).src = variant.wav;
            document.getElementById(`img-recon-${index}`).src = variant.avif;
            
            // Update Info
            const wavKB = (variant.wav_size / 1024).toFixed(2);
            const avifKB = (variant.avif_size / 1024).toFixed(2);
            const ratio = (variant.wav_size / variant.avif_size).toFixed(2);
            
            document.getElementById(`info-${index}`).innerHTML = 
                `Original WAV: ${wavKB} KB<br>` +
                `AVIF Image:   ${avifKB} KB<br>` +
                `<strong>Compression Ratio: ${ratio}x</strong>`;
        }
        
        function updateSize(index, q) {
            // Deprecated, handled in updateSample
        }

        function updateCursor(index, side) {
            const audio = document.getElementById(`audio-${side}-${index}`);
            const cursor = document.getElementById(`cursor-${side}-${index}`);
            const duration = audio.duration;
            const current = audio.currentTime;
            
            if (duration > 0) {
                const percent = (current / duration) * 100;
                cursor.style.left = percent + "%";
            }
        }

        renderSamples();
    </script>
</body>
</html>
    """
    
    import json
    json_data = json.dumps(results)
    final_html = html_content.replace('DATA_PLACEHOLDER', json_data)
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(final_html)

def main():
    parser = argparse.ArgumentParser(description="Compress wav via Mel-Spectrogram AVIF image.")
    parser.add_argument("input", help="Path to wav file or directory containing wav files.")
    parser.add_argument("--output", default="output", help="Output directory.")
    args = parser.parse_args()

    # Setup device and model
    device = get_device()
    print(f"Loading SpeechT5HifiGan on {device}...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()

    # Collect files
    files = []
    if os.path.isfile(args.input):
        files.append(args.input)
    elif os.path.isdir(args.input):
        for root, _, filenames in os.walk(args.input):
            for f in filenames:
                if f.lower().endswith('.wav'):
                    files.append(os.path.join(root, f))
    
    if not files:
        print("No wav files found.")
        return

    os.makedirs(args.output, exist_ok=True)
    
    results = []

    for wav_file in files:
        print(f"Processing {wav_file}...")
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        file_output_dir = os.path.join(args.output, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # 1. Original -> Mel
        try:
            logmel = wav_to_logmel(wav_file) # (T, 80)
        except Exception as e:
            print(f"Error reading {wav_file}: {e}")
            continue

        # Save resampled original for comparison
        # We need the audio data that corresponds to the extracted mel.
        # wav_to_logmel resamples internally but returns mel.
        # Let's just save the resampled audio separately using soundfile/librosa
        wav_orig, sr = librosa.load(wav_file, sr=TARGET_SR, mono=True)
        orig_wav_path = os.path.join(file_output_dir, "original.wav")
        sf.write(orig_wav_path, wav_orig, TARGET_SR)
        orig_wav_size = os.path.getsize(orig_wav_path)
        
        # Save Original Mel as PNG (Lossless)
        img_orig = logmel_to_image(logmel)
        orig_mel_path = os.path.join(file_output_dir, "original_mel.png")
        img_orig.save(orig_mel_path, "PNG")

        variants = {}

        for q in QUALITIES:
            # Mel -> Image
            img = logmel_to_image(logmel)
            
            # Save AVIF
            avif_path = os.path.join(file_output_dir, f"q{q}.avif")
            img.save(avif_path, "AVIF", quality=q)
            avif_size = os.path.getsize(avif_path)
            
            # Load AVIF
            # Note: opening and converting back ensures we see compression artifacts
            img_loaded = Image.open(avif_path)
            
            # Image -> Mel
            logmel_recon = image_to_logmel(img_loaded)
            
            # Mel -> Wav
            wav_recon = reconstruct_wav(logmel_recon, vocoder, device)
            
            # Save Wav
            wav_recon_path = os.path.join(file_output_dir, f"q{q}_recon.wav")
            sf.write(wav_recon_path, wav_recon, TARGET_SR)
            
            # Relative paths for HTML
            variants[str(q)] = {
                'avif': os.path.relpath(avif_path, args.output),
                'wav': os.path.relpath(wav_recon_path, args.output),
                'wav_size': orig_wav_size,
                'avif_size': avif_size
            }
            print(f"  Quality {q}: Saved AVIF and Reconstructed WAV.")

        results.append({
            'name': base_name,
            'original': os.path.relpath(orig_wav_path, args.output),
            'original_mel': os.path.relpath(orig_mel_path, args.output),
            'variants': variants
        })

    generate_html(args.output, results)
    print(f"Done. Open {os.path.join(args.output, 'index.html')} to view results.")

if __name__ == "__main__":
    main()
