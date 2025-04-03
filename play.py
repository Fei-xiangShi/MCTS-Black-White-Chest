import torch
import numpy as np
from board import Board
from model import ReversiNet
from mcts import MCTS
import time
import argparse

class StateHistory:
    def __init__(self, max_length=8):
        self.max_length = max_length
        self.history = []
        
    def add(self, state):
        self.history.append(state.copy())
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    def get_state_with_history(self):
        """返回包含历史的状态张量，用0填充不足的历史"""
        current = self.history[-1]
        channels = []
        
        # 当前黑白
        black_current = (current == 1).astype(float)
        white_current = (current == -1).astype(float)
        channels.extend([black_current, white_current])
        
        # 历史黑白
        for i in range(len(self.history) - 1, -1, -1):
            state = self.history[i]
            black = (state == 1).astype(float)
            white = (state == -1).astype(float)
            channels.extend([black, white])
        
        # 填充不足的历史
        padding_needed = self.max_length - len(self.history) + 1
        for _ in range(padding_needed):
            padding = np.zeros_like(black_current)
            channels.extend([padding, padding])
            
        return np.stack(channels)

class ReversiGame:
    def __init__(self, 
                 model_path: str = "reversi_model_best.pth", 
                 use_mcts: bool = True, 
                 num_simulations: int = 800,
                 history_length: int = 8,
                 device: str = None):
        self.board = Board()
        self.history = StateHistory(max_length=history_length)
        self.history.add(self.board.get_state())
        
        # 确定使用的设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = ReversiNet.load(model_path, history_length=history_length, device=self.device)
        
        self.use_mcts = use_mcts
        self.mcts = MCTS(self.model, num_simulations) if use_mcts else None
        
    def print_board(self):
        """打印当前棋盘状态"""
        print("\n  0 1 2 3 4 5 6 7")
        for i in range(8):
            row = f"{i} "
            for j in range(8):
                if self.board.board[i][j] == Board.EMPTY:
                    row += ". "
                elif self.board.board[i][j] == Board.BLACK:
                    row += "X "
                else:
                    row += "O "
            print(row)
        print()
    
    def get_human_move(self) -> tuple[int, int]:
        """获取人类玩家的落子位置"""
        legal_moves = self.board.get_legal_moves(Board.BLACK)
        print(f"合法落子: {legal_moves}")
        
        while True:
            try:
                row = int(input("请输入行号(0-7): "))
                col = int(input("请输入列号(0-7): "))
                if 0 <= row < 8 and 0 <= col < 8:
                    if (row, col) in legal_moves:
                        return row, col
                    else:
                        print("这不是合法落子位置!")
                else:
                    print("请输入0-7之间的数字!")
            except ValueError:
                print("请输入有效的数字!")
    
    def get_ai_move(self) -> tuple[int, int]:
        """获取AI的落子位置"""
        start_time = time.time()
        
        legal_moves = self.board.get_legal_moves(Board.WHITE)
        if not legal_moves:
            return None
            
        if self.use_mcts:
            # 使用MCTS
            policy = self.mcts.run(self.board, Board.WHITE, self.history)
            
            # 选择概率最高的合法动作
            legal_policy = np.zeros(64)
            for move in legal_moves:
                move_idx = move[0] * 8 + move[1]
                legal_policy[move_idx] = policy[move_idx]
                
            if np.sum(legal_policy) > 0:
                best_move_idx = np.argmax(legal_policy)
                row, col = best_move_idx // 8, best_move_idx % 8
            else:
                # 随机选择
                move_idx = np.random.choice(len(legal_moves))
                row, col = legal_moves[move_idx]
        else:
            # 直接使用策略网络
            state_tensor = torch.tensor(self.history.get_state_with_history(), dtype=torch.float32).unsqueeze(0)
            policy, _ = self.model.predict(state_tensor)
            policy = policy.cpu().numpy().flatten()  # 从GPU获取数据
            
            # 只考虑合法动作
            legal_policy = np.zeros(64)
            for move in legal_moves:
                move_idx = move[0] * 8 + move[1]
                legal_policy[move_idx] = policy[move_idx]
                
            if np.sum(legal_policy) > 0:
                # 取最高概率的动作
                best_move_idx = np.argmax(legal_policy)
                row, col = best_move_idx // 8, best_move_idx % 8
            else:
                # 随机选择
                move_idx = np.random.choice(len(legal_moves))
                row, col = legal_moves[move_idx]
                
        end_time = time.time()
        print(f"AI思考时间: {end_time - start_time:.2f}秒")
        return row, col
    
    def play(self):
        """开始游戏"""
        print("欢迎来到黑白棋游戏!")
        print("你执黑先手(X), AI执白后手(O)")
        print("输入行号和列号(0-7)来落子")
        
        if self.use_mcts:
            print("AI使用MCTS搜索 (较强但较慢)")
        else:
            print("AI直接使用策略网络 (较快但可能较弱)")
        
        current_player = Board.BLACK
        
        while not self.board.is_game_over():
            self.print_board()
            
            if current_player == Board.BLACK:
                # 人类回合
                legal_moves = self.board.get_legal_moves(Board.BLACK)
                if not legal_moves:
                    print("你没有合法落子,跳过回合")
                    current_player = -current_player
                    continue
                    
                row, col = self.get_human_move()
                self.board.make_move(row, col, Board.BLACK)
                self.history.add(self.board.get_state())
                print(f"你落子: ({row}, {col})")
                
            else:
                # AI回合
                legal_moves = self.board.get_legal_moves(Board.WHITE)
                if not legal_moves:
                    print("AI没有合法落子,跳过回合")
                    current_player = -current_player
                    continue
                    
                row, col = self.get_ai_move()
                self.board.make_move(row, col, Board.WHITE)
                self.history.add(self.board.get_state())
                print(f"AI落子: ({row}, {col})")
            
            current_player = -current_player
        
        # 游戏结束
        self.print_board()
        black_count = np.sum(self.board.board == Board.BLACK)
        white_count = np.sum(self.board.board == Board.WHITE)
        print(f"黑棋得分: {black_count}, 白棋得分: {white_count}")
        
        winner = self.board.get_winner()
        if winner == Board.BLACK:
            print("恭喜你获胜!")
        elif winner == Board.WHITE:
            print("AI获胜!")
        else:
            print("平局!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='黑白棋游戏')
    parser.add_argument('--model', type=str, default='reversi_model_best.pth', help='模型文件路径')
    parser.add_argument('--no_mcts', action='store_true', help='不使用MCTS，直接使用策略网络（更快但可能更弱）')
    parser.add_argument('--simulations', type=int, default=800, help='MCTS模拟次数（更多=更强但更慢）')
    parser.add_argument('--device', type=str, default=None, help='使用的设备，可以是"cuda"或"cpu"，默认自动选择')
    parser.add_argument('--gpu_id', type=int, default=0, help='如果有多个GPU，指定使用哪个GPU，默认为0')
    
    args = parser.parse_args()
    
    # 设置GPU设备
    if args.device == 'cuda' and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"发现 {torch.cuda.device_count()} 个 GPU")
            device = f"cuda:{args.gpu_id}"
        else:
            device = "cuda"
        print(f"使用 GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and args.device == 'cuda':
            print("警告: 未找到可用的GPU，使用CPU代替")
    
    game = ReversiGame(
        model_path=args.model,
        use_mcts=not args.no_mcts,
        num_simulations=args.simulations,
        device=device
    )
    game.play() 