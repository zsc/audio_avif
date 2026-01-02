下面说的 **“mel”**，指的是 `microsoft/speecht5_hifigan` 这个 HiFi-GAN vocoder 在 `forward(spectrogram=...)` 里吃进去的那种 **log-mel spectrogram**（时间在前、mel 维在后）。官方文档里也明确它输入的是 **log-mel spectrogram**，形状可以是 `(sequence_length, model_in_dim)` 或 batch 版 `(batch, sequence_length, model_in_dim)`。 ([Hugging Face][1])

---

## 1) `microsoft/speecht5_hifigan` 里 mel 的“定义”到底是什么

在 Transformers 的 `SpeechT5FeatureExtractor` 里，提取 mel 的关键设定（这套设定就是 SpeechT5 / HiFi-GAN 这一套默认对齐的）是：

### 1.1 采样率与时频分辨率（ms → samples）

* **sampling_rate = 16000 Hz** ([Hugging Face][2])
* **win_length = 64 ms**（每帧窗长）([Hugging Face][2])
* **hop_length = 16 ms**（帧移 / shift）([Hugging Face][2])

并且它把毫秒换成采样点的方式是整数除法：

* `sample_size = win_length * sampling_rate // 1000` → `64 * 16000 // 1000 = 1024 samples`
* `sample_stride = hop_length * sampling_rate // 1000` → `16 * 16000 // 1000 = 256 samples` ([Cnblogs][3])

### 1.2 FFT 长度

它用 `optimal_fft_length(sample_size)` 得到 `n_fft`。因为 `sample_size=1024` 已经是 2 的幂，所以 **n_fft=1024**（等价于常见的 1024 点 FFT）。 ([Cnblogs][3])

频点数：`n_freqs = n_fft//2 + 1 = 513`。 ([Cnblogs][3])

### 1.3 窗函数

* **Hann window**（参数名里是 `"hann_window"`）([Hugging Face][2])

### 1.4 Mel 滤波器组（这点很关键）

* **num_mel_bins = 80** ([Hugging Face][2])
* **fmin = 80 Hz**
* **fmax = 7600 Hz** ([Hugging Face][2])
* mel filter bank 的构造里显式指定了：

  * `norm="slaney"`
  * `mel_scale="slaney"` ([Cnblogs][3])

> 也就是说：它不是 HTK mel（默认那套），而是 **Slaney mel + Slaney 归一化**。
> 这会导致你用 librosa/torchaudio 如果没把 `htk/mel_scale/norm` 配对齐，出来的 mel **数值会明显不同**，vocoder/模型效果会被影响。

### 1.5 “log”的定义：不是 ln，也不是 dB，是 **log10**

提取函数里调用 `spectrogram(... log_mel="log10")`： ([Cnblogs][3])

而在 Transformers 的音频工具函数说明里，`log_mel="log10"` 的含义就是 **对谱取以 10 为底的对数**（不是 dB、也不是自然对数）。 ([Hugging Face][4])

另外：

* `mel_floor = 1e-10`（在取对数前把 mel 能量下限截断，避免 log(0)） ([Hugging Face][2])

### 1.6 输出张量的形状（time-first）

`spectrogram()` 算出来的矩阵最后会做一次转置并返回：`return log_mel_spec.T`。 ([Cnblogs][3])

因此最终 mel 的形状是：

* **(num_frames, 80)** 也就是 **(T, 80)**（时间在前，mel bins 在后）

这正好对齐 `SpeechT5HifiGan.forward()` 文档里说的输入形状 `(sequence_length, config.model_in_dim)`（其中 model_in_dim 对应 mel 维，通常就是 80）。 ([Hugging Face][1])

---

## 2) 从 wav 转出它能用的 mel：严格复现版流程

给定 wav（单声道、float），步骤是：

1. **重采样到 16k**（如果不是 16k）
2. **转单声道**（如果是双声道就平均/取一条）
3. STFT：

   * n_fft=1024
   * win_length=1024
   * hop_length=256
   * window=Hann
     -（Transformers 的 `spectrogram` 默认 `center=True, pad_mode="reflect"`；如果你用 librosa/torchaudio，保持默认基本一致即可） ([Hugging Face][4])
4. 幅度谱（power=1.0 的那种；不要用 power=2.0 的 power spectrogram） ([Hugging Face][4])
5. mel filter bank：80 bins，80–7600 Hz，**slaney mel + slaney norm** ([Cnblogs][3])
6. `mel = max(mel, 1e-10)` ([Cnblogs][3])
7. `logmel = log10(mel)`（不是 dB！不是 ln！） ([Cnblogs][3])
8. 转成 **(T, 80)**

---

## 3) 代码示例 A：用 librosa 手动对齐（推荐用于“你自己做 mel”）

```python
import numpy as np
import soundfile as sf
import librosa

def wav_to_speecht5_logmel(
    wav_path: str,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    返回: log-mel (T, 80), float32
    定义对齐 SpeechT5FeatureExtractor:
      - sr=16000
      - win=64ms(1024), hop=16ms(256), n_fft=1024
      - Hann window
      - mel: n_mels=80, fmin=80, fmax=7600, Slaney mel + Slaney norm
      - mel_floor=1e-10
      - log10
    """
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:          # (N, C)
        wav = wav.mean(axis=1) # mono
    wav = wav.astype(np.float32)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    n_fft = 1024
    win_length = 1024
    hop_length = 256

    # 幅度谱（power=1.0 等价于 magnitude）
    S = np.abs(
        librosa.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    )  # (513, T)

    # Slaney mel + Slaney norm
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=80,
        fmin=80,
        fmax=7600,
        htk=False,         # slaney mel（librosa 里 htk=False 对应 slaney 公式）
        norm="slaney",     # area normalization
    )  # (80, 513)

    mel = mel_basis @ S              # (80, T)
    mel = np.maximum(mel, 1e-10)     # mel_floor
    logmel = np.log10(mel)           # log10

    return logmel.T.astype(np.float32)  # (T, 80)
```

**常见踩坑提醒：**

* `librosa.feature.melspectrogram()` 默认 `power=2`（power谱），这里不要用默认；要么自己 stft 取 abs，要么显式 `power=1.0`。
* 不要用 `librosa.power_to_db()` / `amplitude_to_db()`，那是 dB，不是 log10。

---

## 4) 代码示例 B：torchaudio 版（方便直接接到 vocoder）

如果你希望直接产出 `torch.Tensor`：

```python
import torch
import torchaudio

def wav_tensor_to_speecht5_logmel(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    wav: (N,) 或 (1, N) 或 (C, N)
    返回: (T, 80) 的 log10 mel
    """
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # (1, N)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)               # (1, N)

    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        window_fn=torch.hann_window,
        n_mels=80,
        f_min=80.0,
        f_max=7600.0,
        power=1.0,          # 幅度谱
        norm="slaney",
        mel_scale="slaney",
        center=True,
        pad_mode="reflect",
    )(wav)  # (1, 80, T)

    mel_spec = torch.clamp(mel_spec, min=1e-10)
    logmel = torch.log10(mel_spec)          # (1, 80, T)
    return logmel.squeeze(0).transpose(0, 1)  # (T, 80)
```

---

## 5) “它能用”的判定标准

只要你产出的 mel 满足下面几点，就能喂给 `microsoft/speecht5_hifigan`：

* **采样率假设 16k 的那套时频参数**：win=1024, hop=256, n_fft=1024 ([Cnblogs][3])
* **80 维 mel**，频带 **80–7600 Hz** ([Hugging Face][2])
* **Slaney mel + Slaney norm** ([Cnblogs][3])
* **mel_floor=1e-10**，然后 **log10** ([Cnblogs][3])
* shape 为 **(T, 80)**（或 batch 版 `(B, T, 80)`） ([Cnblogs][3])
