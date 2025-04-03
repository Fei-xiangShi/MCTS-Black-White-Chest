import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.query = nn.Conv2d(num_channels, num_channels // 8, 1)
        self.key = nn.Conv2d(num_channels, num_channels // 8, 1)
        self.value = nn.Conv2d(num_channels, num_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        attention = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=1)
        out = torch.bmm(value, attention)
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ReversiNet(nn.Module):
    def __init__(self, board_size: int = 8, num_channels: int = 128, history_length: int = 8, device=None):
        super().__init__()
        self.board_size = board_size
        self.history_length = history_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输入通道: 2 * (history_length + 1) (当前状态和历史状态的黑白棋子位置)
        # 历史长度=8时，通道数应为18: 2*9=18
        # 每个历史状态有2个通道(黑棋和白棋)
        input_channels = 2 * (history_length + 1)
        print(f"Model initialized with input_channels={input_channels}, history_length={history_length}")
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # 残差块和注意力块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(10)
        ])
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(num_channels) for _ in range(5)
        ])
        
        # 策略头 (输出落子概率)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size)
        )
        
        # 价值头 (输出状态价值)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # 移动模型到指定设备
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量, shape为(batch_size, 2 * (history_length + 1), board_size, board_size)
        Returns:
            policy: 落子概率, shape为(batch_size, board_size * board_size)
            value: 状态价值, shape为(batch_size, 1)
        """
        # 确保输入张量在正确的设备上
        x = x.to(self.device)
        features = self.initial_conv(x)
        
        # 应用残差块和注意力块
        for i in range(5):
            features = self.res_blocks[i*2](features)
            features = self.res_blocks[i*2+1](features)
            features = self.attention_blocks[i](features)
        
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
    
    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测给定状态的策略和价值
        Args:
            state: 棋盘状态, shape为(1, 2 * (history_length + 1), board_size, board_size)
        Returns:
            policy: 落子概率
            value: 状态价值
        """
        self.eval()
        with torch.no_grad():
            # 确保输入张量在正确的设备上
            state = state.to(self.device)
            policy, value = self(state)
            policy = F.softmax(policy, dim=1)
        return policy, value
    
    @staticmethod
    def get_device():
        """获取可用的设备（GPU或CPU）"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def save(self, path):
        """保存模型权重"""
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path, board_size=8, history_length=8, device=None):
        """加载模型权重"""
        device = device or cls.get_device()
        model = cls(board_size=board_size, history_length=history_length, device=device)
        # 加载权重到CPU，然后再转移到指定设备上，这样可以在不同设备间迁移模型
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model 