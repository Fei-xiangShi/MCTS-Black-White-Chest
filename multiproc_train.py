import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
import os
from tqdm import tqdm
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from collections import deque

from board import Board
from model import ReversiNet
from mcts import MCTS
from train import StateHistory

class ReversiDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]
    
def self_play_worker(rank, config, result_queue):
    """工作进程执行自对弈任务"""
    # 设置随机种子确保每个进程有不同的随机序列
    np.random.seed(int(time.time() * 1000 + rank) % 2**32)
    random.seed(int(time.time() * 1000 + rank) % 2**32)
    
    # 加载模型
    current_model = ReversiNet(history_length=config['history_length'])
    best_model = ReversiNet(history_length=config['history_length'])
    
    current_model.load_state_dict(torch.load(config['current_model_path']))
    best_model.load_state_dict(torch.load(config['best_model_path']))
    
    current_model.eval()
    best_model.eval()
    
    # 创建MCTS搜索器
    current_mcts = MCTS(current_model, config['num_simulations'])
    best_mcts = MCTS(best_model, config['num_simulations'])
    
    # 根据课程学习阶段调整游戏难度
    temperature = max(0.1, 1.0 - config['curriculum_step'] * 0.2)
    
    # 本进程需要生成的游戏数
    games_per_worker = config['num_games'] // config['num_workers']
    if rank < config['num_games'] % config['num_workers']:
        games_per_worker += 1
    
    training_data = []
    
    for game in range(games_per_worker):
        board = Board(config['board_size'])
        history = StateHistory(max_length=config['history_length'])
        history.add(board.get_state())  # 添加初始状态
        
        player = Board.BLACK
        game_data = []
        
        while not board.is_game_over():
            # 选择使用哪个模型
            if player == Board.BLACK:
                policy = current_mcts.run(board, player, history)
            else:
                policy = best_mcts.run(board, player, history)
            
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
            
            state = history.get_state_with_history()
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
            augmented_data = augment_state(state, policy)
            for aug_state, aug_policy in augmented_data:
                training_data.append((aug_state, aug_policy, value * player_turn))
    
    # 将生成的数据放入队列
    result_queue.put(training_data)

def augment_state(state: np.ndarray, policy: np.ndarray) -> List[tuple]:
    """数据增强：旋转和翻转"""
    augmented_data = []
    
    # 原始状态
    augmented_data.append((state, policy))
    
    # 旋转90度
    rotated_state = np.zeros_like(state)
    for i in range(state.shape[0]):
        rotated_state[i] = np.rot90(state[i], 1)
    
    policy_board = policy.reshape(8, 8)
    rotated_policy = np.rot90(policy_board, 1).flatten()
    augmented_data.append((rotated_state, rotated_policy))
    
    # 旋转180度
    rotated_state = np.zeros_like(state)
    for i in range(state.shape[0]):
        rotated_state[i] = np.rot90(state[i], 2)
    
    policy_board = policy.reshape(8, 8)
    rotated_policy = np.rot90(policy_board, 2).flatten()
    augmented_data.append((rotated_state, rotated_policy))
    
    # 旋转270度
    rotated_state = np.zeros_like(state)
    for i in range(state.shape[0]):
        rotated_state[i] = np.rot90(state[i], 3)
    
    policy_board = policy.reshape(8, 8)
    rotated_policy = np.rot90(policy_board, 3).flatten()
    augmented_data.append((rotated_state, rotated_policy))
    
    # 水平翻转
    flipped_state = np.zeros_like(state)
    for i in range(state.shape[0]):
        flipped_state[i] = np.fliplr(state[i])
    
    policy_board = policy.reshape(8, 8)
    flipped_policy = np.fliplr(policy_board).flatten()
    augmented_data.append((flipped_state, flipped_policy))
    
    return augmented_data

def evaluate_model(current_model_path, best_model_path, config):
    """评估当前模型与最佳模型的对抗表现"""
    current_model = ReversiNet(history_length=config['history_length'])
    best_model = ReversiNet(history_length=config['history_length'])
    
    current_model.load_state_dict(torch.load(current_model_path))
    best_model.load_state_dict(torch.load(best_model_path))
    
    current_model.eval()
    best_model.eval()
    
    # 创建MCTS搜索器
    current_mcts = MCTS(current_model, config['eval_simulations'])
    best_mcts = MCTS(best_model, config['eval_simulations'])
    
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    for game in tqdm(range(config['eval_games']), desc="Evaluating Models"):
        board = Board(config['board_size'])
        history = StateHistory(max_length=config['history_length'])
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
                mcts = current_mcts
            else:
                mcts = best_mcts
            
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
    win_rate = (win_count + 0.5 * draw_count) / config['eval_games']
    print(f"Evaluation Results: Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")
    print(f"Win Rate: {win_rate:.2f}")
    
    return win_rate

def train_model(training_data, current_model_path, config):
    """训练模型"""
    model = ReversiNet(history_length=config['history_length'])
    model.load_state_dict(torch.load(current_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # 随机打乱数据
    random.shuffle(training_data)
    
    # 准备数据
    states, policies, values = [], [], []
    for state, policy, value in training_data:
        states.append(state)
        policies.append(policy)
        values.append(value)
    
    # 创建数据集和数据加载器
    dataset = ReversiDataset(
        np.array(states), 
        np.array(policies), 
        np.array(values)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # 不使用多进程加载数据，避免与主多进程冲突
    )
    
    # 训练模型
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    for states, policies, values in tqdm(dataloader, desc="Training Model"):
        # 转换为张量
        states = states.float()
        policies = policies.float()
        values = values.float()
        
        # 前向传播
        pred_policies, pred_values = model(states)
        
        # 计算损失
        policy_loss = criterion(pred_policies, policies)
        value_loss = value_criterion(pred_values.squeeze(), values)
        loss = policy_loss + value_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * states.size(0)
        total_policy_loss += policy_loss.item() * states.size(0)
        total_value_loss += value_loss.item() * states.size(0)
    
    # 更新学习率
    avg_loss = total_loss / len(dataset)
    scheduler.step(avg_loss)
    
    avg_policy_loss = total_policy_loss / len(dataset)
    avg_value_loss = total_value_loss / len(dataset)
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Policy Loss: {avg_policy_loss:.4f}")
    print(f"Value Loss: {avg_value_loss:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 保存训练后的模型
    model_save_path = f"reversi_model_epoch_{config['epoch']}.pth"
    torch.save(model.state_dict(), model_save_path)
    
    return model_save_path

def main():
    # MP相关设置
    mp.set_start_method('spawn', force=True)
    
    # 配置
    config = {
        'board_size': 8,
        'num_games': 200,  # 每轮自对弈游戏数
        'num_epochs': 100,  # 训练轮数
        'batch_size': 256,
        'learning_rate': 0.001,
        'num_simulations': 200,  # 自对弈时MCTS模拟次数
        'eval_simulations': 400, # 评估时MCTS模拟次数
        'history_length': 8,
        'num_workers': min(8, mp.cpu_count()),  # 工作进程数，最多8个
        'eval_games': 20,
        'win_threshold': 0.55,
        'curriculum_steps': 5,
    }
    
    # 确保模型文件存在
    if not os.path.exists('reversi_model_best.pth'):
        # 初始化模型
        init_model = ReversiNet(history_length=config['history_length'])
        torch.save(init_model.state_dict(), 'reversi_model_best.pth')
    
    # 复制最佳模型作为当前模型
    current_model_path = 'reversi_model_current.pth'
    best_model_path = 'reversi_model_best.pth'
    os.system(f'cp {best_model_path} {current_model_path}')
    
    for epoch in range(config['num_epochs']):
        print(f"\n===== Epoch {epoch + 1}/{config['num_epochs']} =====")
        
        # 课程学习
        curriculum_step = min(epoch // (config['num_epochs'] // config['curriculum_steps']), config['curriculum_steps'] - 1)
        config['curriculum_step'] = curriculum_step
        config['epoch'] = epoch + 1
        
        # 多进程自对弈
        print(f"Self play with {config['num_workers']} workers, generating {config['num_games']} games")
        result_queue = mp.Queue()
        processes = []
        
        config['current_model_path'] = current_model_path
        config['best_model_path'] = best_model_path
        
        for rank in range(config['num_workers']):
            p = mp.Process(target=self_play_worker, args=(rank, config, result_queue))
            p.start()
            processes.append(p)
        
        # 收集数据
        training_data = []
        for _ in range(config['num_workers']):
            worker_data = result_queue.get()
            training_data.extend(worker_data)
        
        # 等待所有进程结束
        for p in processes:
            p.join()
        
        print(f"Collected {len(training_data)} training examples")
        
        # 训练模型
        new_model_path = train_model(training_data, current_model_path, config)
        
        # 评估模型
        win_rate = evaluate_model(new_model_path, best_model_path, config)
        
        # 更新最佳模型
        if win_rate >= config['win_threshold']:
            print(f"New best model found! Win rate: {win_rate:.2f}")
            os.system(f'cp {new_model_path} {best_model_path}')
        else:
            print(f"Current model not better than best model. Win rate: {win_rate:.2f}")
        
        # 更新当前模型
        os.system(f'cp {new_model_path} {current_model_path}')
        
if __name__ == "__main__":
    main() 