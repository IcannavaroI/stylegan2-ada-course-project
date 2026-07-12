# 基于 StyleGAN2-ADA 的小样本图像生成课程设计

本项目是《机器学习与深度学习》课程设计，基于 NVIDIA 官方实现的 StyleGAN2-ADA 框架，在 8GB 显存的硬件限制下，使用 MetFaces 数据集完成了小样本图像生成模型的训练与参数调优实验。

## 📁 项目概述
- **研究目标**：验证 ADA 机制在小样本（1336张图像）场景下的有效性，并探索硬件受限条件下的最佳训练配置。
- **主要成果**：实现了稳定的模型训练，最佳配置下 FID 达到 49.16，并对比分析了批大小、R1 正则化系数等超参数的影响。
- **技术栈**：Python 3.8, PyTorch 1.9.0, CUDA 11.1, StyleGAN2-ADA

## 📊 数据集
本项目使用 **MetFaces 数据集**（大都会博物馆艺术人脸画像），共 1,336 张图像，原始分辨率为 1024x1024。为适配 8GB 显存，预处理时已降采样至 256x256。

## ⚙️ 环境配置与安装

### 系统要求
- **操作系统**：Ubuntu 20.04 LTS（或 Windows 11 + WSL 2）
- **GPU**：NVIDIA GPU 8GB 显存（如 RTX 2070）
- **CUDA**：11.1 或更新版本
- **Python**：3.8

### 安装步骤
1. **克隆本仓库**
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

2. **创建并激活 Conda 环境**
conda create -n stylegan2 python=3.8 -y
conda activate stylegan2

3. **安装 PyTorch 和项目依赖**
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

4. **（可选）安装 TensorBoard 以查看训练曲线**
pip install tensorboard

🚀 快速开始
1. 数据准备
将你的图片放入 ./datasets/my_dataset/ 文件夹，然后运行以下命令生成 .zip 格式的数据集：
python dataset_tool.py --source=./datasets/my_dataset --dest=./datasets/my_dataset.zip --width=256 --height=256 --transform=center-crop

2. 模型训练
使用优化后的配置启动训练（推荐 batch size = 16）：
python train.py --outdir=./training-runs \
                --data=./datasets/metfaces_256.zip \
                --gpus=1 \
                --cfg=paper256 \
                --aug=ada \
                --batch=16 \
                --gamma=10 \
                --metrics=none

3. 生成图像
使用训练好的模型生成图像：
python generate.py --outdir=out --trunc=1 --seeds=0-49 --network=./training-runs/你的训练目录/network-snapshot-000000.pkl

4. 计算 FID 指标
python calc_metrics.py --metrics=fid50k --network=./training-runs/你的训练目录/network-snapshot-000600.pkl --data=./datasets/metfaces_256.zip

📈 实验结果摘要
最优配置	   Batch Size：16	   显存占用：~4.8 GB	   sec/kimg：~57.7	        FID@600kimg：49.16
更详细的实验设计与分析请参阅 课程设计报告。

📝 引用
本项目基于以下论文和代码仓库：

Karras, T., et al. "Training Generative Adversarial Networks with Limited Data." NeurIPS 2020.

NVIDIA StyleGAN2-ADA PyTorch Implementation

📄 许可证
本项目仅供学习与课程设计使用，相关代码版权归 NVIDIA 所有。
