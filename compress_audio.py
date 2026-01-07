import argparse
import os
import soundfile as sf
import librosa
from PIL import Image
import audio_avif

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
        <h1>Audio Compression via Image Mel-Spectrogram</h1>
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
                            <div class="spectrogram-container" id="spec-container-orig-${index}">
                                <img src="${sample.original_mel}" class="spectrogram-img"> 
                                <div id="cursor-orig-${index}" class="cursor"></div>
                            </div>
                        </div>
                        
                        <div class="panel">
                            <h3>Reconstructed (from <span id="lbl-q-${index}">${defaultQ}</span>)</h3>
                            <audio id="audio-recon-${index}" controls src="${sample.variants[defaultQ].wav}" ontimeupdate="updateCursor(${index}, 'recon')"></audio>
                            <div class="spectrogram-container" id="spec-container-recon-${index}">
                                <img id="img-recon-${index}" src="${sample.variants[defaultQ].image}" class="spectrogram-img">
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
            document.getElementById(`img-recon-${index}`).src = variant.image;
            
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
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(final_html)

def decode_file(input_img, output_wav, device, vocoder):
    print(f"Decoding {input_img} -> {output_wav}...")
    try:
        if input_img.lower().endswith('.webp'):
            logmel, rms = audio_avif.webp_anim_to_logmel(input_img)
        else:
            img = Image.open(input_img)
            logmel, rms = audio_avif.image_to_logmel(img)
            
        wav = audio_avif.reconstruct_wav(logmel, vocoder, device)
        
        if rms is not None:
            print(f"  Restoring loudness (RMS: {rms:.4f})...")
            wav = audio_avif.apply_loudness(wav, rms)
            
        sf.write(output_wav, wav, audio_avif.TARGET_SR)
        print(f"Success. Saved to {output_wav}")
    except Exception as e:
        print(f"Error decoding {input_img}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compress/Decompress audio via Mel-Spectrogram image (AVIF or JPEG).")
    parser.add_argument("input", help="Input file (WAV, AVIF, JPEG, WEBP) or directory (WAVs).")
    parser.add_argument("--output", default=None, help="Output directory (for batch/demo) or filename (for single file). Defaults to 'output' for batch.")
    parser.add_argument("--jpg", action="store_true", help="Use JPEG instead of AVIF for compression.")
    parser.add_argument("--png", action="store_true", help="Use PNG (Lossless) instead of AVIF.")
    parser.add_argument("--sq", action="store_true", help="Enable square-reshaping heuristic. Default is linear (long strip).")
    parser.add_argument("--webp-video", action="store_true", help="Enable WebP animation (pseudo-video) compression experiment.")
    args = parser.parse_args()

    # Determine mode based on input extension
    input_ext = os.path.splitext(args.input)[1].lower()
    is_decoding = os.path.isfile(args.input) and input_ext in ['.avif', '.jpg', '.jpeg', '.webp', '.png']
    
    # Setup device and model
    device = audio_avif.get_device()
    print(f"Loading SpeechT5HifiGan on {device}...")
    vocoder = audio_avif.load_vocoder(device)

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
             
        decode_file(args.input, output_path, device, vocoder)
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

    for wav_file in files:
        print(f"Processing {wav_file}...")
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        file_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # 1. Original -> Mel
        try:
            logmel, rms = audio_avif.wav_to_logmel(wav_file) # (T, 80)
        except Exception as e:
            print(f"Error reading {wav_file}: {e}")
            continue

        # Save resampled original for comparison
        wav_orig, sr = librosa.load(wav_file, sr=audio_avif.TARGET_SR, mono=True)
        orig_wav_path = os.path.join(file_output_dir, "original.wav")
        sf.write(orig_wav_path, wav_orig, audio_avif.TARGET_SR)
        orig_wav_size = os.path.getsize(orig_wav_path)
        
        # Save Original Mel as PNG (Lossless) - ALWAYS Linear (reshape=False) for visualization
        img_orig = audio_avif.logmel_to_image(logmel, rms=rms, reshape=False) # Embed RMS
        orig_mel_path = os.path.join(file_output_dir, "original_mel.png")
        img_orig.save(orig_mel_path, "PNG", exif=img_orig.getexif()) # Explicitly save Exif

        variants = {}

        # Standard AVIF/JPEG/PNG Loop
        for q in qualities:
            # Mel -> Image
            img = audio_avif.logmel_to_image(logmel, rms=rms, reshape=use_square)
            
            # Save Compressed Image
            img_path = os.path.join(file_output_dir, f"q{q}.{img_ext}")
            
            if img_format == "PNG":
                img.save(img_path, img_format, exif=img.getexif())
                key = "PNG Lossless"
            else:
                img.save(img_path, img_format, quality=q, exif=img.getexif())
                key = f"{img_format} Q{q}"

            compressed_img_size = os.path.getsize(img_path)
            
            # Load Compressed Image
            img_loaded = Image.open(img_path)
            
            # Image -> Mel
            logmel_recon, rms_recon = audio_avif.image_to_logmel(img_loaded)
            
            # Mel -> Wav
            wav_recon = audio_avif.reconstruct_wav(logmel_recon, vocoder, device)
            
            # Restore Loudness
            if rms_recon:
                wav_recon = audio_avif.apply_loudness(wav_recon, rms_recon)
            
            # Save Wav
            wav_recon_path = os.path.join(file_output_dir, f"q{q}_recon.wav")
            sf.write(wav_recon_path, wav_recon, audio_avif.TARGET_SR)
            
            # Use distinct keys
            key = f"{img_format} Q{q}"
            variants[key] = {
                'image': os.path.relpath(img_path, output_dir),
                'wav': os.path.relpath(wav_recon_path, output_dir),
                'wav_size': orig_wav_size,
                'image_size': compressed_img_size
            }
            print(f"  {key}: Saved and Reconstructed.")

        # Optional WebP Loop
        if use_webp:
            for q in audio_avif.QUALITIES:
                webp_path = os.path.join(file_output_dir, f"webp_video_q{q}.webp")
                
                # Encode (this handles chunking internally)
                # Chunk width 128 (approx 2s)
                audio_avif.logmel_to_webp_anim(logmel, rms, webp_path, quality=q, chunk_width=128)
                compressed_img_size = os.path.getsize(webp_path)
                
                # Decode
                logmel_recon, rms_recon = audio_avif.webp_anim_to_logmel(webp_path)
                wav_recon = audio_avif.reconstruct_wav(logmel_recon, vocoder, device)
                if rms_recon:
                    wav_recon = audio_avif.apply_loudness(wav_recon, rms_recon)
                    
                wav_recon_path = os.path.join(file_output_dir, f"webp_video_q{q}_recon.wav")
                sf.write(wav_recon_path, wav_recon, audio_avif.TARGET_SR)
                
                key = f"WebP-Video Q{q}"
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
            'variants': variants
        })

    generate_html(output_dir, results)
    print(f"Done. Open {os.path.join(output_dir, 'index.html')} to view results.")

if __name__ == "__main__":
    main()