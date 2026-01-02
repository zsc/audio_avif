我们希望实现一个wav 的经过图像的压缩解压。方法如下，输入 wav 单声道，提取 mel spectrum （一张灰度图，定义对齐 mel.md）。这张图可以进行 avif 压缩（压缩质量可调）。解压时再逆变换出 mel, 再经microsoft/speecht5\_hifigan 转回 wav。

输入 wav （或指定目录，一个目录下多个 wav）在 python 命令行指定，默认生成avif 质量 70, 80, 85, 90, 95 几种结果。输出的 html 上，可以 drop down 显示哪种 avif 质量的 side-by-side 显示 wav 和重建 wav，并可以直接播放压缩前后的 wav. 播放 wav 时在 mel-spectrum 上有 cursor 表示播放进度。
