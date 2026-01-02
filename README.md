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

### 2.2 图像映射与压缩 (Mel $\to$ AVIF)
- **数值映射**: 将 Log-Mel 谱的浮点数值（范围约为 -11.0 到 4.0）线性映射到 0-255 的 8-bit 灰度空间。
    - 时间轴映射为图像的**宽度**。
    - 频率轴映射为图像的**高度**（低频在下，高频在上）。
- **编码**: 使用 `pillow-avif-plugin` 将灰度图编码为 AVIF 格式。
    - **Quality**: 支持 70, 80, 85, 90, 95 等不同质量因子，以此控制比特率。

### 2.3 音频重建 (AVIF $\to$ WAV)
- **解码**: 读取 AVIF 图像，逆映射回浮点 Log-Mel 谱。
- **声码器**: 使用预训练的 HiFi-GAN 模型 (`microsoft/speecht5_hifigan`) 将 Mel 谱还原为时域波形。

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

## 5. 运行指南

```bash
# 安装依赖
pip install torch torchaudio transformers librosa soundfile pillow pillow-avif-plugin numpy scipy

# 运行压缩脚本
python3 compress_audio.py input.wav --output results_dir
```

运行结束后，打开 `results_dir/index.html` 即可在浏览器中对比试听和查看压缩比数据。

```