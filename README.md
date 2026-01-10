# 基于图像压缩的音频编码实验

本项目探索一种非传统的音频压缩方法：将音频转换为时频图（Mel Spectrogram），利用现代图像编码格式（AVIF）的高效压缩能力对谱图进行压缩，最后通过神经声码器（HiFi-GAN）将解码后的图像重建为音频。

## 1. 背景

传统音频编码（如 MP3, AAC, Opus）主要依赖心理声学模型去除人耳听觉掩蔽效应下的冗余信息。而随着生成式 AI 的发展，神经声码器（Neural Vocoder）展示了从低维特征（如 Mel 谱）高质量重建波形的能力。

本项目旨在验证以下链路的可行性：
> **音频波形 $\rightarrow$ 视觉谱图 $\rightarrow$ 图像压缩 (AVIF) $\rightarrow$ 视觉谱图 $\rightarrow$ 神经声码器 $\rightarrow$ 重建音频**

通过这种方式，我们将“时序信号压缩”问题转化为了“图像压缩”问题，利用 AVIF (基于 AV1 视频编码) 强大的帧内预测和量化能力来压缩声学特征。

## 2. 算法流程

整个处理管线包含以下步骤：

### 2.1 特征提取 (WAV $\to$ Mel)
为了适配 `microsoft/speecht5_hifigan` 声码器，我们严格遵循其训练时的特征定义：
- **采样率**: 16000 Hz (单声道)
- **STFT 参数**: FFT size 1024, Window size 1024, Hop size 256 (16ms)
- **窗函数**: Hann window
- **Mel 滤波器**: 80 bands, 频率范围 80Hz - 7600Hz
- **关键细节**: 使用 **Slaney** 风格的 Mel 刻度和归一化 (Area Normalization)，取 $log_{10}$ 幅度谱 (with floor 1e-10)。

### 2.2 图像映射与压缩 (Mel $\to$ Image)
- **数值映射**: 将 Log-Mel 谱的浮点数值（范围约为 -11.0 到 4.0）线性映射到 0-255 的 8-bit 灰度空间。
    - 时间轴映射为图像的**宽度**。
    - 频率轴映射为图像的**高度**（低频在下，高频在上）。
- **重排 (Reshaping)**: 可选功能。为了提高图像编码效率，可通过 `--sq` 参数将细长的频谱图切片并堆叠成接近方形的图像（宽度按 16 对齐）。解码时根据元数据自动还原。
- **水平拉伸 (Stretching)**: 可选功能。通过 `--stretch` 参数（如 `--stretch 2.0`）在宽度方向对频谱图进行重采样。由于解码时不会自动缩放回原长，这会直接改变重建音频的时长（如拉伸 2 倍会导致音频播放速度减慢一倍）。该选项可用于实验不同像素密度下的编码表现，或作为一种简单的变速手段。
- **水平去模糊 (Unsharp Masking)**: 可选功能。通过 `--horizontal-usm <kernel-size,sigma,strength>` 参数（如 `--horizontal-usm 10,3,0.1`）在宽度方向（时间轴）应用一维反锐化掩模。这可以用于在编码前补偿因水平拉伸或有损压缩导致的细节模糊，增强时间轴方向的特征对比度。
- **伪视频模式 (Pseudo-Video)**: 通过 `--webp-video` 参数启用。将长频谱图沿时间轴切分为多个小块（默认约 2 秒/块），并将这些块作为连续帧保存为 **WebP 动图**。这利用了 WebP 对帧间差异的压缩能力（尽管频谱图的“帧间”相关性与普通视频不同），在某些情况下能提供优于单张大图的压缩率。
- **元数据嵌入**: 计算原始音频的 RMS (Root Mean Square) 振幅，并作为元数据（Exif Tag 270 ImageDescription）嵌入到图像中。这确保了解码后的音频能还原到原始响度。
- **编码**: 默认使用 `pillow-avif-plugin` 将灰度图编码为 AVIF 格式，也支持使用 JPEG 和 WebP (动图) 格式。
    - **Quality**: 支持 70, 80, 85, 90, 95 等不同质量因子，以此控制比特率。

### 2.3 音频重建 (Image $\to$ WAV)
- **解码**: 读取图像（AVIF, JPEG 或 WebP），逆映射回浮点 Log-Mel 谱。同时读取 Exif 元数据中的 RMS 值。
- **声码器**: 使用预训练的 HiFi-GAN 模型 (`microsoft/speecht5_hifigan`) 将 Mel 谱还原为时域波形。
- **响度恢复**: 根据读取的 RMS 值对重建波形进行增益调整，使其响度与原始音频一致。

## 3. 注意事项

1.  **参数对齐至关重要**:
    HiFi-GAN 对输入的 Mel 谱特征非常敏感。如果计算 Mel 谱时使用了错误的参数（如使用了 HTK 风格而非 Slaney，或者使用了 `ln` 而非 `log10`），重建出的音频会有严重的金属音或噪声。本项目已严格校准至 SpeechT5 标准。

2.  **量化误差**:
    当前方案将 float32 的 Mel 谱量化为 8-bit 整数保存为图像。这种量化本身会引入底噪，但在 AVIF 有损压缩的大幅失真面前，8-bit 量化噪声通常不是主要瓶颈。

3.  **动态范围截断**:
    我们将 Log-Mel 的值域固定在 `[-11.0, 4.0]` 进行归一化。如果输入音频极其微弱或超出此范围，可能会导致截断失真（Clipping）。

4.  **环境依赖**:
    需要安装 `pillow-avif-plugin` 才能让 PIL 支持 AVIF 格式。macOS 上安装时可能需要注意 `libavif` 的系统库依赖。

## 4. Future Work

*   **高位深图像支持**: 探索使用 10-bit 或 12-bit 的 AVIF/HEIF 格式，以减少将 Log-Mel 转为像素时的量化损失。
*   **自适应动态范围**: 不使用固定的 `[-11, 4]` 范围，而是将每段音频的 Min/Max 值作为元数据（Metadata）存储在图像头部，实现无损的动态范围映射。
*   **模型微调 (Fine-tuning)**: 现有的 HiFi-GAN 是在干净的 Mel 谱上训练的。可以在经过 AVIF 压缩/解压后的“有损 Mel 谱”上微调声码器，使其学会修复压缩伪影（Artifacts），从而在低码率下获得更好的听感。
*   **其他图像编码**: 对比 WebP, JPEG XL, HEIC 在频谱图压缩上的表现。
*   **多通道支持**: 将立体声或多声道音频映射到图像的 RGB 通道进行压缩。

## 5. 安装

本项目已支持通过 `pip` 安装：

```bash
# 在项目根目录下执行
pip install .

# 或者以编辑模式安装
pip install -e .
```

## 6. 使用指南

### 6.1 命令行工具 (CLI)

安装后，可以直接使用 `audio-avif` 命令处理音频文件：

1.  **压缩/演示模式**:
    处理单个 WAV 文件或整个目录，生成多种质量的图像和 HTML 对比报告。
    ```bash
    # 默认生成 AVIF
    audio-avif input.wav --output results_dir
    
    # 使用 JPEG
    audio-avif input.wav --output results_dir --jpg

    # 启用方形重排 (Square Reshaping)
    audio-avif input.wav --output results_dir --sq

    # 启用水平拉伸 (如 2 倍拉伸，会使重建音频变慢一倍)
    audio-avif input.wav --output results_dir --stretch 2.0

    # 启用水平去模糊 (Unsharp Mask)
    audio-avif input.wav --output results_dir --horizontal-usm 10,3,0.1

    # 启用 WebP 动图模式 (Pseudo-Video)
    audio-avif input.wav --output results_dir --webp-video
    ```

2.  **解码模式**:
    脚本会根据文件扩展名自动识别输入格式（`.avif`, `.jpg`, `.jpeg`, `.webp`）。
    ```bash
    # 将 input.avif 解码为 input.wav
    audio-avif input.avif
    
    # 将 input.jpg 解码为 output.wav
    audio-avif input.jpg --output output.wav

    # 将 input.webp 解码为 output.wav
    audio-avif input.webp --output output.wav
    ```

### 6.2 Python API (作为编解码库使用)

你也可以在代码中直接调用 `audio_avif` 提供的核心功能：

```python
import audio_avif
from PIL import Image
import soundfile as sf

# 1. 准备环境 (加载声码器)
device = audio_avif.get_device()
vocoder = audio_avif.load_vocoder(device)

# 2. 编码: WAV -> AVIF
# 提取 Log-Mel 谱
logmel = audio_avif.wav_to_logmel("input.wav")
# 转换为图像并保存为 AVIF
img = audio_avif.logmel_to_image(logmel)
img.save("compressed.avif", "AVIF", quality=85)

# 3. 解码: AVIF -> WAV
# 加载 AVIF 图像
img_loaded = Image.open("compressed.avif")
# 还原 Log-Mel 谱
logmel_recon = audio_avif.image_to_logmel(img_loaded)
# 使用声码器重建音频
wav_recon = audio_avif.reconstruct_wav(logmel_recon, vocoder, device)

# 4. 保存结果
sf.write("reconstructed.wav", wav_recon, audio_avif.TARGET_SR)
```

## 7. 测试



本项目包含单元测试，用于验证编解码流程的连通性及基本信号质量（PSNR）。测试过程中会自动进行响度对齐，并将对齐参数保存到 JSON 文件中。



运行测试：

```bash

python3 -m unittest discover tests

```



## 8. 注意事项
