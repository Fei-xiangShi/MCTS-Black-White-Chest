# 黑白棋 AI 系统

基于深度强化学习和蒙特卡洛树搜索（MCTS）的黑白棋 AI 实现，采用类似 AlphaZero 的架构设计。

## 功能特点

- **强大的 AI 引擎**：结合策略网络和 MCTS 搜索的混合架构
- **自我学习**：通过自对弈不断提升棋力
- **多种优化**：
  - 残差网络和注意力机制
  - 历史状态记忆
  - 模型评估与淘汰机制
  - 数据增强
  - 课程学习
- **多进程训练**：充分利用多核 CPU 加速训练
- **GPU 加速**：支持 CUDA 加速训练和推理
- **灵活配置**：可根据需要调整各项参数
- **友好界面**：简单易用的命令行工具

## 安装要求

- Python 3.7+
- PyTorch 1.7+（支持 CUDA 的版本更佳）
- NumPy
- tqdm

可以通过以下命令安装依赖：

```bash
# CPU 版本
pip install torch numpy tqdm

# GPU 版本 (CUDA 11.7 示例)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy tqdm
```

## 快速开始

### 训练模型

```bash
# 单进程训练 (CPU)
python run.py train --mode single --games 100 --epochs 10 --simulations 200

# 单进程训练 (GPU)
python run.py train --mode single --games 100 --epochs 10 --simulations 200 --gpu_id 0

# 多进程训练（更快）
python run.py train --mode multi --games 200 --epochs 50 --simulations 200

# 多进程训练 (GPU)
python run.py train --mode multi --games 200 --epochs 50 --simulations 200 --gpu_id 0
```

### 与 AI 对战

```bash
# 使用 MCTS（更强但更慢）- CPU
python run.py play --model reversi_model_best.pth --simulations 800

# 使用 MCTS - GPU
python run.py play --model reversi_model_best.pth --simulations 800 --gpu_id 0

# 直接使用策略网络（更快但可能较弱）- GPU
python run.py play --model reversi_model_best.pth --no_mcts --gpu_id 0
```

### 评估模型

```bash
# 比较两个模型的强度 (CPU)
python run.py eval --model1 reversi_model_best.pth --model2 reversi_model_epoch_10.pth --games 20

# 比较两个模型的强度 (GPU)
python run.py eval --model1 reversi_model_best.pth --model2 reversi_model_epoch_10.pth --games 20 --gpu_id 0
```

## 系统架构

### 主要组件

- **模型架构**（`model.py`）：结合残差网络和注意力机制的深度神经网络
- **MCTS 搜索**（`mcts.py`）：基于策略网络引导的蒙特卡洛树搜索
- **棋盘逻辑**（`board.py`）：黑白棋规则实现
- **训练系统**：
  - 单进程版（`train.py`）
  - 多进程版（`multiproc_train.py`）
- **游戏接口**（`play.py`）：人机对战界面
- **命令行工具**（`run.py`）：统一的操作入口

### 训练策略

1. **双模型互博**：当前模型与"最佳"模型对弈生成训练数据
2. **广泛数据增强**：通过旋转、翻转等操作增加数据多样性
3. **经验回放**：保留历史经验避免"灾难性遗忘"
4. **课程学习**：从高探索到低探索，逐步提高挑战难度
5. **评估机制**：只有真正变强的模型才会被保存为"最佳"模型

## 参数说明

### 训练参数

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `--mode` | 训练模式（单进程/多进程） | single | single/multi |
| `--games` | 每轮自对弈游戏数 | 100 | 100-1000 |
| `--epochs` | 训练轮数 | 10 | 10-100 |
| `--simulations` | MCTS 模拟次数 | 200 | 100-800 |
| `--batch_size` | 训练批次大小 | 32/256 | 32-512 |
| `--no_gpu` | 不使用 GPU | False | - |
| `--gpu_id` | 使用的 GPU ID | 0 | 0-(GPU数-1) |

### 游戏参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | reversi_model_best.pth |
| `--no_mcts` | 不使用 MCTS（直接用策略网络） | False |
| `--simulations` | MCTS 模拟次数 | 800 |
| `--no_gpu` | 不使用 GPU | False |
| `--gpu_id` | 使用的 GPU ID | 0 |

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model1` | 模型1文件路径 | reversi_model_best.pth |
| `--model2` | 模型2文件路径 | reversi_model_current.pth |
| `--games` | 评估游戏数量 | 20 |
| `--simulations` | MCTS 模拟次数 | 400 |
| `--no_gpu` | 不使用 GPU | False |
| `--gpu_id` | 使用的 GPU ID | 0 |

## 训练建议

1. **初期训练**：
   - 使用较小的游戏数（100-200）
   - 较少的 MCTS 模拟次数（100-200）
   - 快速迭代产生初步有效的模型
   - CPU 训练足够

2. **中期训练**：
   - 增加游戏数（300-500）
   - 增加 MCTS 模拟次数（400-600）
   - 使用多进程加速训练
   - GPU 加速更有效

3. **后期优化**：
   - 大量游戏数（500+）
   - 较多的 MCTS 模拟次数（600+）
   - 调低学习率进行精细调整
   - GPU 加速几乎是必需的

## GPU 加速

系统支持 GPU 加速训练和推理，以显著提高性能：

- **多 GPU 支持**：如果系统有多个 GPU，可以通过 `--gpu_id` 参数指定使用哪个 GPU
- **多进程训练**：在多进程训练中，系统会自动将不同进程分配到不同 GPU
- **内存优化**：模型会自动管理 GPU 内存使用，包括在需要时将数据迁移到 GPU

要获得最佳性能，推荐：
- 使用支持 CUDA 的 GPU（NVIDIA）
- 安装与 GPU 兼容的 PyTorch 版本
- 训练时使用较大的批次大小（对于 GPU 训练，推荐 256-512）

## 技术细节

### 模型架构

- 多通道输入：棋盘当前状态 + 历史状态
- 主干网络：残差块 + 注意力机制
- 双头输出：
  - 策略头：预测落子概率
  - 价值头：评估当前局面价值

### MCTS 实现

- 虚拟损失：支持并行搜索
- UCB 公式：平衡探索与利用
- 根据历史记录预测行为

## 问题排查

1. **训练过慢**：
   - 减少 MCTS 模拟次数
   - 使用多进程训练
   - 开启 GPU 加速

2. **GPU 内存不足**：
   - 减小批次大小
   - 减少模型通道数
   - 考虑使用混合精度训练

3. **AI 太弱**：
   - 增加训练轮数
   - 增加 MCTS 模拟次数
   - 检查训练损失是否稳定下降

4. **模型无法加载**：
   - 确保模型结构与保存时一致
   - 检查 PyTorch 版本兼容性
   - 检查 CPU/GPU 兼容性问题

## 致谢

本项目实现受到 AlphaZero/AlphaGo 系列论文的启发，并借鉴了多种深度强化学习技术。

## 许可证

MIT License
