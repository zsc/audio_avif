import argparse
import os
import soundfile as sf
import librosa
from PIL import Image
import audio_avif

def get_decoder_description(decoder, griffin_lim_iters, mel_bins):
    if decoder == "griffin-lim":
        return (
            f"Using librosa mel inversion + Griffin-Lim "
            f"({griffin_lim_iters} iterations) for waveform reconstruction "
            f"with {mel_bins}-bin log-mel."
        )
    return (
        f"Using microsoft/speecht5_hifigan for vocoding with {mel_bins}-bin log-mel."
    )

def generate_html(output_dir, results, decoder_description):
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
        <h1>Audio Compression via Image Mel-Spectrogram</h1>
        <p>DECODER_DESCRIPTION_PLACEHOLDER</p>
        
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
                const keys = Object.keys(sample.variants);
                const defaultQ = keys.includes("AVIF Q90") ? "AVIF Q90" : keys[0]; 
                
                div.innerHTML = `
                    <h2>${sample.name}</h2>
                    <div class="controls">
                        <label>Variant: 
                            <select id="sel-${index}" onchange="updateSample(${index})">
                                ${keys.map(q => `<option value="${q}" ${q === defaultQ ? 'selected' : ''}>${q}</option>`).join('')}
                            </select>
                        </label>
                    </div>
                    
                    <div class="comparison">
                        <div class="panel">
                            <h3>Original (Resampled 16kHz)</h3>
                            <audio id="audio-orig-${index}" controls src="${sample.original}" ontimeupdate="updateCursor(${index}, 'orig')"></audio>
                            <div style="font-size: 12px; color: #555; margin-top: 6px;">Mel Spectrum</div>
                            <div class="spectrogram-container" id="spec-container-orig-${index}">
                                <img src="${sample.original_mel}" class="spectrogram-img"> 
                                <div id="cursor-orig-${index}" class="cursor"></div>
                            </div>
                            ${sample.original_mfcc ? `
                                <div style="font-size: 12px; color: #555; margin-top: 10px;">MFCC (Encoding)</div>
                                <div class="spectrogram-container">
                                    <img src="${sample.original_mfcc}" class="spectrogram-img">
                                </div>
                            ` : ''}
                        </div>
                        
                        <div class="panel">
                            <h3>Reconstructed (from <span id="lbl-q-${index}">${defaultQ}</span>)</h3>
                            <audio id="audio-recon-${index}" controls src="${sample.variants[defaultQ].wav}" ontimeupdate="updateCursor(${index}, 'recon')"></audio>
                            <div style="font-size: 12px; color: #555; margin-top: 6px;">Encoded Image</div>
                            <div class="spectrogram-container" id="spec-container-recon-${index}">
                                <img id="img-recon-${index}" src="${sample.variants[defaultQ].image}" class="spectrogram-img">
                                <div id="cursor-recon-${index}" class="cursor"></div>
                            </div>
                            ${sample.variants[defaultQ].recon_mel ? `
                                <div style="font-size: 12px; color: #555; margin-top: 10px;">Reconstructed Mel</div>
                                <div class="spectrogram-container">
                                    <img id="img-recon-mel-${index}" src="${sample.variants[defaultQ].recon_mel}" class="spectrogram-img">
                                </div>
                            ` : ''}
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
            document.getElementById(`img-recon-${index}`).src = variant.image;
            const reconMelImg = document.getElementById(`img-recon-mel-${index}`);
            if (reconMelImg && variant.recon_mel) {
                reconMelImg.src = variant.recon_mel;
            }
            
            // Update Info
            const wavKB = (variant.wav_size / 1024).toFixed(2);
            const imageKB = (variant.image_size / 1024).toFixed(2);
            const ratio = (variant.wav_size / variant.image_size).toFixed(2);
            
            document.getElementById(`info-${index}`).innerHTML = 
                `Original WAV: ${wavKB} KB<br>` +
                `Compressed:   ${imageKB} KB<br>` +
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
    final_html = final_html.replace('DECODER_DESCRIPTION_PLACEHOLDER', decoder_description)
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(final_html)

def decode_file(
    input_img,
    output_wav,
    decoder,
    griffin_lim_iters,
    mel_bins,
    device=None,
    vocoder=None,
    is_mfcc=False,
):
    print(f"Decoding {input_img} -> {output_wav}...")
    try:
        if input_img.lower().endswith('.webp'):
            logmel, rms = audio_avif.webp_anim_to_logmel(input_img)
        else:
            img = Image.open(input_img)
            data, meta = audio_avif.image_to_matrix(img)
            rms = meta.get('rms')
            
            # Check metadata for type
            if meta.get('type') == 'mfcc' or is_mfcc:
                print("  Interpreting as MFCC...")
                target_mel_bins = meta.get('mel_bins') or mel_bins
                logmel = audio_avif.mfcc_to_logmel(data, n_mels=target_mel_bins)
            else:
                logmel = data
            
        wav = audio_avif.reconstruct_wav(
            logmel,
            vocoder=vocoder,
            device=device,
            decoder=decoder,
            griffin_lim_iters=griffin_lim_iters,
        )
        
        if rms is not None:
            print(f"  Restoring loudness (RMS: {rms:.4f})...")
            wav = audio_avif.apply_loudness(wav, rms)
            
        sf.write(output_wav, wav, audio_avif.TARGET_SR)
        print(f"Success. Saved to {output_wav}")
    except Exception as e:
        print(f"Error decoding {input_img}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Compress/Decompress audio via Mel-Spectrogram image (AVIF or JPEG).")
    parser.add_argument("input", help="Input file (WAV, AVIF, JPEG, WEBP) or directory (WAVs).")
    parser.add_argument("--output", default=None, help="Output directory (for batch/demo) or filename (for single file). Defaults to 'output' for batch.")
    parser.add_argument("--jpg", action="store_true", help="Use JPEG instead of AVIF for compression.")
    parser.add_argument("--png", action="store_true", help="Use PNG (Lossless) instead of AVIF.")
    parser.add_argument("--mfcc", type=int, default=None, metavar="HEIGHT",
                        help="Use MFCC features with given height (n_mfcc) instead of Mel-Spectrogram.")
    parser.add_argument(
        "--mel-bins",
        type=int,
        default=audio_avif.N_MELS,
        help="Number of mel bins for mel extraction/reconstruction. SpeechT5 vocoder currently requires 80; Griffin-Lim supports other values such as 128.",
    )
    parser.add_argument("--sq", action="store_true", help="Enable square-reshaping heuristic. Default is linear (long strip).")
    parser.add_argument("--stretch", type=float, default=1.0, help="Horizontal stretch factor (e.g. 2.0 for 2x width, 0.5 for 0.5x width).")
    parser.add_argument("--webp-video", action="store_true", help="Enable WebP animation (pseudo-video) compression experiment.")
    parser.add_argument("--horizontal-gaussian", type=str, default=None, help="Add 1D horizontal Gaussian blur, e.g. '10,3' for kernel-size=10, sigma=3.")
    parser.add_argument("--horizontal-usm", type=str, default=None, help="Add 1D horizontal unsharp mask, e.g. '10,3,0.1' for kernel-size=10, sigma=3, strength=0.1.")
    parser.add_argument("--shift-key", type=str, default="0", help="Shift key (pitch) in pixels, comma-separated. E.g. '0' or '5,-5'.")
    parser.add_argument(
        "--decoder",
        choices=audio_avif.DECODER_CHOICES,
        default="vocoder",
        help="Waveform reconstruction backend. 'vocoder' uses SpeechT5 HiFi-GAN; 'griffin-lim' avoids the neural vocoder.",
    )
    parser.add_argument(
        "--griffin-lim-iters",
        type=int,
        default=audio_avif.DEFAULT_GRIFFIN_LIM_ITERS,
        help="Number of Griffin-Lim iterations when --decoder=griffin-lim.",
    )
    args = parser.parse_args()

    if args.mel_bins <= 0:
        parser.error("--mel-bins must be a positive integer.")

    # Parse Shift Keys
    try:
        shift_keys = [int(x.strip()) for x in args.shift_key.split(',')]
    except ValueError:
        print("Invalid --shift-key format. Use comma-separated integers like '5,-5,0'.")
        return

    # Parse Gaussian Blur
    gaussian_blur = None
    if args.horizontal_gaussian:
        try:
            ks, sig = map(float, args.horizontal_gaussian.split(','))
            gaussian_blur = (ks, sig)
        except ValueError:
            print("Invalid --horizontal-gaussian format. Use 'kernel_size,sigma'.")
            return

    # Parse USM
    horizontal_usm = None
    if args.horizontal_usm:
        try:
            ks, sig, strg = map(float, args.horizontal_usm.split(','))
            horizontal_usm = (ks, sig, strg)
        except ValueError:
            print("Invalid --horizontal-usm format. Use 'kernel_size,sigma,strength'.")
            return

    # Determine mode based on input extension
    input_ext = os.path.splitext(args.input)[1].lower()
    is_decoding = os.path.isfile(args.input) and input_ext in ['.avif', '.jpg', '.jpeg', '.webp', '.png']
    if input_ext == ".avif" and not audio_avif.HAS_AVIF:
        parser.error("AVIF input requires pillow-avif-plugin. Install it or decode a JPEG/WebP instead.")
    if not is_decoding and not args.jpg and not args.png and not audio_avif.HAS_AVIF:
        parser.error("AVIF output requires pillow-avif-plugin. Install it or rerun with --jpg or --png.")
    
    device = None
    vocoder = None
    if args.decoder == "vocoder":
        if args.mel_bins != audio_avif.N_MELS:
            parser.error(
                f"--decoder=vocoder currently requires --mel-bins={audio_avif.N_MELS}."
            )
        device = audio_avif.get_device()
        print(f"Loading SpeechT5HifiGan on {device}...")
        vocoder = audio_avif.load_vocoder(device)
    else:
        print(
            f"Using Griffin-Lim decoder "
            f"({args.griffin_lim_iters} iterations, no SpeechT5 vocoder)."
        )

    # --- DECODE MODE ---
    if is_decoding:
        output_path = args.output
        if output_path is None:
            # Default to input filename with .wav extension
            output_path = os.path.splitext(args.input)[0] + ".wav"
        
        # If user provided a directory as output, place file inside
        if os.path.isdir(output_path) or (args.output and not os.path.splitext(args.output)[1]):
             os.makedirs(output_path, exist_ok=True)
             output_path = os.path.join(output_path, os.path.splitext(os.path.basename(args.input))[0] + ".wav")
             
        decode_file(
            args.input,
            output_path,
            decoder=args.decoder,
            griffin_lim_iters=args.griffin_lim_iters,
            mel_bins=args.mel_bins,
            device=device,
            vocoder=vocoder,
            is_mfcc=args.mfcc is not None,
        )
        return

    # --- ENCODE/DEMO MODE (Existing Logic) ---
    # Handle default output dir if not specified
    output_dir = args.output if args.output else "output"

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

    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    # Format settings
    if args.png:
        img_ext = "png"
        img_format = "PNG"
        qualities = [0] # Dummy quality for loop
    elif args.jpg:
        img_ext = "jpg"
        img_format = "JPEG"
        qualities = audio_avif.QUALITIES
    else:
        img_ext = "avif"
        img_format = "AVIF"
        qualities = audio_avif.QUALITIES

    use_square = args.sq
    use_webp = args.webp_video
    use_mfcc = args.mfcc is not None
    mfcc_height = args.mfcc

    for wav_file in files:
        print(f"Processing {wav_file}...")
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        file_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # 1. Original -> Data
        try:
            if use_mfcc:
                # Use MFCC for encoding, but keep Mel for display
                data_orig, rms = audio_avif.wav_to_mfcc(
                    wav_file,
                    n_mfcc=mfcc_height,
                    n_mels=args.mel_bins,
                )
                data_type = 'mfcc'
                logmel_display, _ = audio_avif.wav_to_logmel(
                    wav_file,
                    n_mels=args.mel_bins,
                )
            else:
                # Use LogMel
                data_orig, rms = audio_avif.wav_to_logmel(
                    wav_file,
                    n_mels=args.mel_bins,
                )
                data_type = 'mel'
                logmel_display = data_orig
        except Exception as e:
            print(f"Error reading {wav_file}: {e}")
            continue

        # Save resampled original for comparison
        wav_orig, sr = librosa.load(wav_file, sr=audio_avif.TARGET_SR, mono=True)
        orig_wav_path = os.path.join(file_output_dir, "original.wav")
        sf.write(orig_wav_path, wav_orig, audio_avif.TARGET_SR)
        orig_wav_size = os.path.getsize(orig_wav_path)
        
        # Save Original Mel as PNG (Lossless) - ALWAYS Linear (reshape=False) for visualization
        # We pass explicit metadata for consistency
        img_orig_mel = audio_avif.matrix_to_image(
            logmel_display,
            metadata={'rms': rms, 'type': 'mel', 'mel_bins': args.mel_bins},
            reshape=False,
        )
        orig_mel_path = os.path.join(file_output_dir, "original_mel.png")
        img_orig_mel.save(orig_mel_path, "PNG", exif=img_orig_mel.getexif()) # Explicitly save Exif

        # If using MFCC, also save MFCC image for reference
        orig_mfcc_path = None
        if use_mfcc:
            img_orig_mfcc = audio_avif.matrix_to_image(
                data_orig,
                metadata={'rms': rms, 'type': 'mfcc', 'mel_bins': args.mel_bins},
                reshape=False,
            )
            orig_mfcc_path = os.path.join(file_output_dir, "original_mfcc.png")
            img_orig_mfcc.save(orig_mfcc_path, "PNG", exif=img_orig_mfcc.getexif())

        variants = {}

        # Standard AVIF/JPEG/PNG Loop
        for q in qualities:
            # Data -> Image (Sequential Shifts)
            # We must pass the SAME metadata to ensure decoding works (min/max are computed inside if not passed, which is fine)
            img = audio_avif.matrix_to_image(
                data_orig,
                metadata={'rms': rms, 'type': data_type, 'mel_bins': args.mel_bins},
                reshape=use_square,
                stretch=args.stretch,
                gaussian_blur=gaussian_blur,
                horizontal_usm=horizontal_usm,
                shift_key=shift_keys,
            )
            
            # Save Compressed Image
            # Suffix shows all shifts if any
            if all(s == 0 for s in shift_keys):
                shift_suffix = ""
            else:
                shift_suffix = "_s" + "_".join(map(str, shift_keys))

            img_path = os.path.join(file_output_dir, f"q{q}{shift_suffix}.{img_ext}")
            
            if img_format == "PNG":
                img.save(img_path, img_format, exif=img.getexif())
                base_key = "PNG Lossless"
            else:
                img.save(img_path, img_format, quality=q, exif=img.getexif())
                base_key = f"{img_format} Q{q}"

            compressed_img_size = os.path.getsize(img_path)
            
            # Load Compressed Image
            img_loaded = Image.open(img_path)
            
            # Image -> Data
            data_recon, meta_recon = audio_avif.image_to_matrix(img_loaded)
            rms_recon = meta_recon.get('rms')
            
            # Data -> Wav
            if use_mfcc or meta_recon.get('type') == 'mfcc':
                target_mel_bins = meta_recon.get('mel_bins') or args.mel_bins
                logmel_recon = audio_avif.mfcc_to_logmel(
                    data_recon,
                    n_mels=target_mel_bins,
                )
            else:
                logmel_recon = data_recon
            
            # Mel -> Wav
            wav_recon = audio_avif.reconstruct_wav(
                logmel_recon,
                vocoder=vocoder,
                device=device,
                decoder=args.decoder,
                griffin_lim_iters=args.griffin_lim_iters,
            )
            
            # Restore Loudness
            if rms_recon is not None:
                wav_recon = audio_avif.apply_loudness(wav_recon, rms_recon)
            
            # Save Wav
            wav_recon_path = os.path.join(file_output_dir, f"q{q}{shift_suffix}_recon.wav")
            sf.write(wav_recon_path, wav_recon, audio_avif.TARGET_SR)

            # Save reconstructed Mel image for display (MFCC mode)
            recon_mel_path = None
            if use_mfcc or meta_recon.get('type') == 'mfcc':
                img_recon_mel = audio_avif.matrix_to_image(
                    logmel_recon,
                    metadata={
                        'rms': rms_recon,
                        'type': 'mel',
                        'mel_bins': logmel_recon.shape[1],
                    },
                    reshape=False,
                )
                recon_mel_path = os.path.join(file_output_dir, f"q{q}{shift_suffix}_recon_mel.png")
                img_recon_mel.save(recon_mel_path, "PNG", exif=img_recon_mel.getexif())
            
            # Use distinct keys
            if all(s == 0 for s in shift_keys):
                key = base_key
            else:
                key = f"Shifted({','.join(map(str, shift_keys))}) {base_key}"

            variants[key] = {
                'image': os.path.relpath(img_path, output_dir),
                'recon_mel': os.path.relpath(recon_mel_path, output_dir) if recon_mel_path else None,
                'wav': os.path.relpath(wav_recon_path, output_dir),
                'wav_size': orig_wav_size,
                'image_size': compressed_img_size
            }
            print(f"  {key}: Saved and Reconstructed.")

        # Optional WebP Loop (Experimental - Keeping as Mel for now unless I update it properly, but user might assume it follows --mfcc)
        # For safety, I will skip WebP if use_mfcc is True to avoid confusion or errors, or default to Mel.
        # But 'logmel_to_webp_anim' uses 'logmel'. If I pass MFCC data to it, it might work but 'webp_anim_to_logmel' will assume it returns logmel.
        # I'll disable WebP for MFCC for now or warning.
        if use_webp:
            if use_mfcc:
                print("Warning: WebP Video experiment currently supports Mel-spectrogram only. Skipping.")
            else:
                for q in audio_avif.QUALITIES:
                    if all(s == 0 for s in shift_keys):
                        shift_suffix = ""
                    else:
                        shift_suffix = "_s" + "_".join(map(str, shift_keys))

                    webp_path = os.path.join(file_output_dir, f"webp_video_q{q}{shift_suffix}.webp")
                    
                    # Encode (Sequential Shifts)
                    audio_avif.logmel_to_webp_anim(
                        logmel_display,
                        rms,
                        webp_path,
                        quality=q,
                        chunk_width=128,
                        gaussian_blur=gaussian_blur,
                        horizontal_usm=horizontal_usm,
                        shift_key=shift_keys,
                    )
                    compressed_img_size = os.path.getsize(webp_path)
                    
                    # Decode
                    logmel_recon, rms_recon = audio_avif.webp_anim_to_logmel(webp_path)
                    wav_recon = audio_avif.reconstruct_wav(
                        logmel_recon,
                        vocoder=vocoder,
                        device=device,
                        decoder=args.decoder,
                        griffin_lim_iters=args.griffin_lim_iters,
                    )
                    if rms_recon is not None:
                        wav_recon = audio_avif.apply_loudness(wav_recon, rms_recon)
                        
                    wav_recon_path = os.path.join(file_output_dir, f"webp_video_q{q}{shift_suffix}_recon.wav")
                    sf.write(wav_recon_path, wav_recon, audio_avif.TARGET_SR)
                    
                    if all(s == 0 for s in shift_keys):
                        key = f"WebP-Video Q{q}"
                    else:
                        key = f"Shifted({','.join(map(str, shift_keys))}) WebP-Video Q{q}"

                    variants[key] = {
                        'image': os.path.relpath(webp_path, output_dir),
                        'wav': os.path.relpath(wav_recon_path, output_dir),
                        'wav_size': orig_wav_size,
                        'image_size': compressed_img_size
                    }
                    print(f"  {key}: Saved and Reconstructed.")

        results.append({
            'name': base_name,
            'original': os.path.relpath(orig_wav_path, output_dir),
            'original_mel': os.path.relpath(orig_mel_path, output_dir),
            'original_mfcc': os.path.relpath(orig_mfcc_path, output_dir) if orig_mfcc_path else None,
            'variants': variants
        })

    generate_html(
        output_dir,
        results,
        get_decoder_description(args.decoder, args.griffin_lim_iters, args.mel_bins),
    )
    print(f"Done. Open {os.path.join(output_dir, 'index.html')} to view results.")

if __name__ == "__main__":
    main()
