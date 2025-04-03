import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Deque
from collections import deque
import random
import os
import argparse
from board import Board
from model import ReversiNet
from mcts import MCTS

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

class ReversiTrainer:
    def __init__(self, 
                 board_size: int = 8,
                 num_simulations: int = 800,
                 num_games: int = 100,
                 num_epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 replay_buffer_size: int = 10000,
                 curriculum_steps: int = 5,
                 history_length: int = 8,
                 eval_games: int = 20,
                 win_threshold: float = 0.55):
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.curriculum_steps = curriculum_steps
        self.history_length = history_length
        self.eval_games = eval_games
        self.win_threshold = win_threshold
        
        # 创建当前模型和最佳模型
        self.current_model = ReversiNet(board_size, history_length=history_length)
        self.best_model = ReversiNet(board_size, history_length=history_length)
        
        # 加载最佳模型如果存在
        if os.path.exists('reversi_model_best.pth'):
            self.best_model.load_state_dict(torch.load('reversi_model_best.pth'))
        
        self.current_model.load_state_dict(self.best_model.state_dict())
        
        # MCTS和优化器
        self.current_mcts = MCTS(self.current_model, num_simulations)
        self.best_mcts = MCTS(self.best_model, num_simulations)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        self.criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
    def _prepare_state(self, history: StateHistory) -> np.ndarray:
        """将带历史的棋盘状态转换为模型输入格式"""
        return history.get_state_with_history()
    
    def _augment_state(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """数据增强：旋转和翻转"""
        augmented_data = []
        
        # 原始状态
        augmented_data.append((state, policy))
        
        # 旋转90度
        rotated_state = np.zeros_like(state)
        rotated_policy = np.zeros_like(policy)
        
        # 对每个通道进行旋转
        for i in range(state.shape[0]):
            rotated_state[i] = np.rot90(state[i], 1)
        
        # 对策略进行旋转
        policy_board = policy.reshape(8, 8)
        rotated_policy = np.rot90(policy_board, 1).flatten()
        augmented_data.append((rotated_state, rotated_policy))
        
        # 旋转180度
        rotated_state = np.zeros_like(state)
        rotated_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            rotated_state[i] = np.rot90(state[i], 2)
        
        policy_board = policy.reshape(8, 8)
        rotated_policy = np.rot90(policy_board, 2).flatten()
        augmented_data.append((rotated_state, rotated_policy))
        
        # 旋转270度
        rotated_state = np.zeros_like(state)
        rotated_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            rotated_state[i] = np.rot90(state[i], 3)
        
        policy_board = policy.reshape(8, 8)
        rotated_policy = np.rot90(policy_board, 3).flatten()
        augmented_data.append((rotated_state, rotated_policy))
        
        # 水平翻转
        flipped_state = np.zeros_like(state)
        flipped_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            flipped_state[i] = np.fliplr(state[i])
        
        policy_board = policy.reshape(8, 8)
        flipped_policy = np.fliplr(policy_board).flatten()
        augmented_data.append((flipped_state, flipped_policy))
        
        # 垂直翻转
        flipped_state = np.zeros_like(state)
        flipped_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            flipped_state[i] = np.flipud(state[i])
        
        policy_board = policy.reshape(8, 8)
        flipped_policy = np.flipud(policy_board).flatten()
        augmented_data.append((flipped_state, flipped_policy))
        
        # 对角线翻转
        flipped_state = np.zeros_like(state)
        flipped_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            flipped_state[i] = np.transpose(state[i])
        
        policy_board = policy.reshape(8, 8)
        flipped_policy = np.transpose(policy_board).flatten()
        augmented_data.append((flipped_state, flipped_policy))
        
        # 反对角线翻转
        flipped_state = np.zeros_like(state)
        flipped_policy = np.zeros_like(policy)
        
        for i in range(state.shape[0]):
            flipped_state[i] = np.rot90(np.transpose(state[i]), 2)
        
        policy_board = policy.reshape(8, 8)
        flipped_policy = np.rot90(np.transpose(policy_board), 2).flatten()
        augmented_data.append((flipped_state, flipped_policy))
        
        return augmented_data
        
    def self_play(self, curriculum_step: int = 0) -> None:
        """
        新旧模型互博生成训练数据，支持课程学习
        """
        # 根据课程学习阶段调整游戏难度
        temperature = max(0.1, 1.0 - curriculum_step * 0.2)
        
        for game in tqdm(range(self.num_games), desc="Self Play"):
            board = Board(self.board_size)
            history = StateHistory(max_length=self.history_length)
            history.add(board.get_state())  # 添加初始状态
            
            player = Board.BLACK
            game_data = []
            
            while not board.is_game_over():
                # 选择使用哪个模型
                if player == Board.BLACK:
                    policy = self.current_mcts.run(board, player, history)
                else:
                    policy = self.best_mcts.run(board, player, history)
                
                legal_moves = board.get_legal_moves(player)
                
                if not legal_moves:
                    player = -player
                    continue
                
                # 应用温度参数
                policy = np.power(policy, 1.0/temperature)
                policy = policy / np.sum(policy)
                
                # 将非法动作的概率设为0
                mask = np.zeros(64)
                for move in legal_moves:
                    mask[move[0] * 8 + move[1]] = 1
                policy = policy * mask
                
                if np.sum(policy) > 0:
                    policy = policy / np.sum(policy)
                else:
                    policy = mask / np.sum(mask)
                
                state = self._prepare_state(history)
                game_data.append((state, policy, player))
                
                # 选择动作
                if len(legal_moves) > 0:
                    if np.sum(policy[mask == 1]) > 0:
                        probs = policy[mask == 1] / np.sum(policy[mask == 1])
                        move_idx = np.random.choice(len(legal_moves), p=probs)
                        row, col = legal_moves[move_idx]
                    else:
                        # 如果所有概率为0，随机选择
                        move_idx = np.random.choice(len(legal_moves))
                        row, col = legal_moves[move_idx]
                    
                    board.make_move(row, col, player)
                    history.add(board.get_state())  # 添加新状态到历史
                    player = -player
            
            # 计算游戏结果
            winner = board.get_winner()
            if winner is not None:
                value = 1 if winner == Board.BLACK else -1
            else:
                value = 0
                
            # 添加训练数据
            for state, policy, player_turn in game_data:
                # 数据增强
                augmented_data = self._augment_state(state, policy)
                for aug_state, aug_policy in augmented_data:
                    self.replay_buffer.append((aug_state, aug_policy, value * player_turn))
    
    def evaluate_models(self) -> float:
        """
        评估当前模型与最佳模型的对抗表现
        返回当前模型的胜率
        """
        win_count = 0
        loss_count = 0
        draw_count = 0
        
        for game in tqdm(range(self.eval_games), desc="Evaluating Models"):
            board = Board(self.board_size)
            history = StateHistory(max_length=self.history_length)
            history.add(board.get_state())
            
            # 交替先手，确保公平
            if game % 2 == 0:
                current_player = Board.BLACK  # 当前模型先手
                best_player = Board.WHITE     # 最佳模型后手
            else:
                current_player = Board.WHITE  # 当前模型后手
                best_player = Board.BLACK     # 最佳模型先手
                
            player = Board.BLACK  # 游戏从黑棋开始
            
            while not board.is_game_over():
                legal_moves = board.get_legal_moves(player)
                
                if not legal_moves:
                    player = -player
                    continue
                
                # 根据当前玩家选择模型
                if player == current_player:
                    mcts = self.current_mcts
                else:
                    mcts = self.best_mcts
                
                # 获取策略
                policy = mcts.run(board, player, history)
                
                # 选择最佳动作（评估时不需要探索）
                legal_policy = np.zeros(64)
                for move in legal_moves:
                    move_idx = move[0] * 8 + move[1]
                    legal_policy[move_idx] = policy[move_idx]
                
                if np.sum(legal_policy) > 0:
                    best_move_idx = np.argmax(legal_policy)
                    row, col = best_move_idx // 8, best_move_idx % 8
                else:
                    # 随机选择合法动作
                    move_idx = np.random.choice(len(legal_moves))
                    row, col = legal_moves[move_idx]
                
                # 执行动作
                board.make_move(row, col, player)
                history.add(board.get_state())
                player = -player
            
            # 游戏结束，计算结果
            winner = board.get_winner()
            if winner is None:  # 平局
                draw_count += 1
            elif (winner == Board.BLACK and current_player == Board.BLACK) or (winner == Board.WHITE and current_player == Board.WHITE):
                win_count += 1  # 当前模型赢
            else:
                loss_count += 1  # 当前模型输
        
        # 计算胜率（计入平局的一半）
        win_rate = (win_count + 0.5 * draw_count) / self.eval_games
        print(f"Evaluation Results: Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")
        print(f"Win Rate: {win_rate:.2f}")
        
        return win_rate
    
    def train(self):
        """训练模型"""
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # 课程学习
            curriculum_step = min(epoch // (self.num_epochs // self.curriculum_steps), self.curriculum_steps - 1)
            
            # 生成训练数据
            self.self_play(curriculum_step)
            
            # 训练模型
            self.current_model.train()
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            
            # 随机打乱数据
            training_data = list(self.replay_buffer)
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    continue  # 跳过不完整的批次
                
                states, policies, values = zip(*batch)
                
                # 准备批次数据
                states = torch.tensor(np.stack(states), dtype=torch.float32)
                policies = torch.tensor(np.stack(policies), dtype=torch.float32)
                values = torch.tensor(values, dtype=torch.float32)
                
                # 前向传播
                pred_policies, pred_values = self.current_model(states)
                
                # 计算损失
                policy_loss = self.criterion(pred_policies, policies)
                value_loss = self.value_criterion(pred_values.squeeze(), values)
                loss = policy_loss + value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
            
            # 更新学习率
            avg_loss = total_loss / (len(training_data) / self.batch_size)
            self.scheduler.step(avg_loss)
            
            avg_policy_loss = total_policy_loss / (len(training_data) / self.batch_size)
            avg_value_loss = total_value_loss / (len(training_data) / self.batch_size)
            
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Policy Loss: {avg_policy_loss:.4f}")
            print(f"Value Loss: {avg_value_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存当前模型
            torch.save(self.current_model.state_dict(), f"reversi_model_epoch_{epoch + 1}.pth")
            
            # 评估当前模型
            if epoch > 0:  # 跳过第一轮评估，因为模型刚初始化
                win_rate = self.evaluate_models()
                
                # 如果当前模型比最佳模型强，则更新最佳模型
                if win_rate >= self.win_threshold:
                    print(f"New best model found! Win rate: {win_rate:.2f}")
                    self.best_model.load_state_dict(self.current_model.state_dict())
                    torch.save(self.best_model.state_dict(), "reversi_model_best.pth")
                else:
                    print(f"Current model not better than best model. Win rate: {win_rate:.2f}")

def parse_args():
    parser = argparse.ArgumentParser(description='训练黑白棋AI')
    parser.add_argument('--num_games', type=int, default=100, help='每轮自对弈游戏数')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--num_simulations', type=int, default=800, help='MCTS模拟次数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = ReversiTrainer(
        num_games=args.num_games,
        num_epochs=args.num_epochs,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    trainer.train() 