import os
import torch
from model import ReversiNet

def reset_model():
    print("清理旧模型文件...")
    # 删除所有reversi模型文件
    for file in os.listdir('.'):
        if file.startswith('reversi_model') and file.endswith('.pth'):
            os.remove(file)
            print(f"已删除: {file}")
    
    print("\n创建新的模型文件...")
    # 创建新的模型文件
    history_length = 8  # 确保与StateHistory类中设置一致
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建新模型
    model = ReversiNet(history_length=history_length, device=device)
    
    # 保存模型
    model.save('reversi_model_best.pth')
    print(f"已创建新模型: reversi_model_best.pth (历史长度={history_length})")
    
    # 显示模型通道数信息
    input_channels = 2 * (history_length + 1)
    print(f"模型输入通道数: {input_channels}")
    print("确保StateHistory类中的实现与此匹配")
    
    print("\n重置完成!")
    print("现在可以运行训练脚本: python run.py train --mode multi --games 100 --epochs 10 --simulations 200 --gpu_id 0")

if __name__ == "__main__":
    reset_model() 