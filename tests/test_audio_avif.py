import unittest
import os
import json
import numpy as np
import soundfile as sf
import audio_avif
from PIL import Image

class TestAudioAvif(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.wav_path = os.path.join(self.test_dir, "test_tone.wav")
        self.alignment_json_path = os.path.join(self.test_dir, "loudness_alignment.json")
        
        # Generate a synthetic sine wave (440Hz) for 1 second
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        sf.write(self.wav_path, audio, sr)

    def tearDown(self):
        if os.path.exists(self.wav_path):
            os.remove(self.wav_path)
        if os.path.exists(self.alignment_json_path):
            os.remove(self.alignment_json_path)
        if os.path.exists(self.test_dir):
            try:
                import shutil
                shutil.rmtree(self.test_dir)
            except:
                pass

    def calculate_psnr(self, original, reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr

    def load_vocoder_or_skip(self):
        try:
            device = audio_avif.get_device()
            vocoder = audio_avif.load_vocoder(device)
        except Exception as e:
            self.skipTest(f"Failed to load vocoder: {e}")
        return device, vocoder

    def estimate_peak_frequencies(self, waveform, sample_rate, top_k=20):
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
        if len(spectrum) > 0:
            spectrum[0] = 0.0
        peak_indices = np.argsort(spectrum)[-top_k:]
        return freqs[peak_indices]

    def test_compression_cycle_psnr(self):
        self.device, self.vocoder = self.load_vocoder_or_skip()

        temp_ext = ".avif" if audio_avif.HAS_AVIF else ".jpg"
        temp_format = "AVIF" if audio_avif.HAS_AVIF else "JPEG"

        # 1. Wav -> Mel (with RMS)
        logmel, rms_orig = audio_avif.wav_to_logmel(self.wav_path)
        
        # 2. Mel -> Image (lossy image, embed RMS)
        img = audio_avif.logmel_to_image(logmel, rms=rms_orig)
        temp_image = os.path.join(self.test_dir, f"temp{temp_ext}")
        img.save(temp_image, temp_format, quality=90, exif=img.getexif())
        
        # 3. Image -> Mel (Extract RMS)
        img_loaded = Image.open(temp_image)
        logmel_recon, rms_recon = audio_avif.image_to_logmel(img_loaded)
        
        self.assertIsNotNone(rms_recon, "RMS metadata should be recovered from image metadata")
        self.assertAlmostEqual(rms_orig, rms_recon, places=4, msg="Recovered RMS should match original")
        
        # 4. Mel -> Wav
        wav_recon = audio_avif.reconstruct_wav(logmel_recon, self.vocoder, self.device)
        
        # Apply loudness correction
        wav_recon_aligned = audio_avif.apply_loudness(wav_recon, rms_recon)
        
        # Load original for comparison
        wav_orig, _ = sf.read(self.wav_path)
        
        # --- Stats for JSON ---
        rms_final = np.sqrt(np.mean(wav_recon_aligned**2))
        
        alignment_info = {
            "original_rms": float(rms_orig),
            "reconstructed_rms_before_align": float(np.sqrt(np.mean(wav_recon**2))),
            "reconstructed_rms_after_align": float(rms_final),
            "metadata_rms": float(rms_recon) if rms_recon is not None else None,
            "note": "RMS stored in Exif/ImageDescription and applied."
        }
        
        with open(self.alignment_json_path, 'w') as f:
            json.dump(alignment_info, f, indent=4)
            
        psnr = self.calculate_psnr(wav_orig, wav_recon_aligned)
        
        print(f"\nTest Result - PSNR: {psnr:.2f} dB")
        print(f"Alignment Metadata Check: Orig={rms_orig:.4f}, Recon={rms_recon:.4f}")
        
        os.remove(temp_image)

        self.assertTrue(psnr > 0, "PSNR should be positive")

    def test_griffin_lim_reconstruction_cycle(self):
        logmel, rms_orig = audio_avif.wav_to_logmel(self.wav_path)
        img = audio_avif.logmel_to_image(logmel, rms=rms_orig)
        logmel_recon, rms_recon = audio_avif.image_to_logmel(img)

        wav_recon = audio_avif.reconstruct_wav(
            logmel_recon,
            decoder="griffin-lim",
            griffin_lim_iters=12,
        )
        wav_recon_aligned = audio_avif.apply_loudness(wav_recon, rms_recon)

        self.assertTrue(np.isfinite(wav_recon_aligned).all(), "Waveform should not contain NaN/Inf")
        self.assertGreater(len(wav_recon_aligned), 0, "Waveform should not be empty")
        self.assertGreater(np.sqrt(np.mean(wav_recon_aligned**2)), 1e-3, "Waveform should contain audible energy")

        peaks = self.estimate_peak_frequencies(wav_recon_aligned, audio_avif.TARGET_SR)
        self.assertTrue(np.any(np.abs(peaks - 440.0) < 35.0), "Should retain a peak near 440 Hz")
        self.assertTrue(np.any(np.abs(peaks - 880.0) < 45.0), "Should retain a peak near 880 Hz")

    def test_reshape_logic(self):
        # Generate longer audio (5s) to trigger reshaping
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        long_wav_path = os.path.join(self.test_dir, "long_tone.wav")
        sf.write(long_wav_path, audio, sr)
        
        logmel, rms_orig = audio_avif.wav_to_logmel(long_wav_path)
        T_orig = logmel.shape[0] # (T, 80)
        
        # 1. Test WITH Reshaping (Explicit)
        img_reshaped = audio_avif.logmel_to_image(logmel, rms=rms_orig, reshape=True)
        w, h = img_reshaped.size
        
        # Expect roughly square
        # T ~ 313 frames. 80*313 = 25040 pixels. sqrt ~ 158.
        # k = round(sqrt(313/80)) = round(1.97) = 2.
        # Height should be 80 * 2 = 160.
        self.assertEqual(h % 80, 0, "Height should be multiple of 80")
        self.assertTrue(h > 80, "Should have stacked at least 2 strips for 5s audio")
        
        # Check metadata
        exif = img_reshaped.getexif()
        desc = exif.get(270)
        self.assertIn("orig_w", desc, "Metadata should contain original width")
        
        # Reconstruction
        logmel_recon, rms_recon = audio_avif.image_to_logmel(img_reshaped)
        self.assertEqual(logmel_recon.shape, logmel.shape, "Reconstructed shape must match original")
        
        # 2. Test WITHOUT Reshaping
        img_linear = audio_avif.logmel_to_image(logmel, rms=rms_orig, reshape=False)
        w_lin, h_lin = img_linear.size
        self.assertEqual(h_lin, 80, "Linear height must be 80")
        self.assertEqual(w_lin, T_orig, "Linear width must match time frames")
        
        # Reconstruction
        logmel_recon_lin, _ = audio_avif.image_to_logmel(img_linear)
        self.assertEqual(logmel_recon_lin.shape, logmel.shape)
        
        os.remove(long_wav_path)

if __name__ == '__main__':
    unittest.main()
