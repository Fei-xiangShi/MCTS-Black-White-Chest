#!/usr/bin/env python
import argparse
import os
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description='黑白棋 AI - 训练与游戏')

    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='选择要执行的操作')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练AI模型')
    train_parser.add_argument(
        '--mode', choices=['single', 'multi'], default='single', help='训练模式：单进程(single)或多进程(multi)')
    train_parser.add_argument(
        '--games', type=int, default=100, help='每轮自对弈游戏数')
    train_parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    train_parser.add_argument(
        '--simulations', type=int, default=200, help='MCTS模拟次数')
    train_parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    train_parser.add_argument('--no_gpu', action='store_true', help='不使用GPU加速')
    train_parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID，如果有多个GPU')

    # 游戏命令
    play_parser = subparsers.add_parser('play', help='与AI对战')
    play_parser.add_argument(
        '--model', type=str, default='reversi_model_best.pth', help='模型文件路径')
    play_parser.add_argument(
        '--no_mcts', action='store_true', help='不使用MCTS，直接使用策略网络（更快但可能更弱）')
    play_parser.add_argument('--simulations', type=int,
                             default=800, help='MCTS模拟次数（更多=更强但更慢）')
    play_parser.add_argument('--no_gpu', action='store_true', help='不使用GPU加速')
    play_parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID，如果有多个GPU')

    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--model1', type=str,
                             default='reversi_model_best.pth', help='模型1文件路径')
    eval_parser.add_argument(
        '--model2', type=str, default='reversi_model_current.pth', help='模型2文件路径')
    eval_parser.add_argument('--games', type=int, default=20, help='评估游戏数量')
    eval_parser.add_argument('--simulations', type=int,
                             default=400, help='MCTS模拟次数')
    eval_parser.add_argument('--no_gpu', action='store_true', help='不使用GPU加速')
    eval_parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID，如果有多个GPU')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 检查GPU可用性
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_info = f"发现 GPU: {torch.cuda.get_device_name(0)}"
        if torch.cuda.device_count() > 1:
            gpu_info += f" (共 {torch.cuda.device_count()} 个GPU)"
        print(gpu_info)
    else:
        print("未检测到可用的 GPU，将使用 CPU 运行")

    # 执行相应命令
    if args.command == 'train':
        gpu_flag = "" if args.no_gpu else "--device cuda"
        gpu_id_flag = f"--gpu_id {args.gpu_id}" if has_gpu and not args.no_gpu else ""
        
        if args.mode == 'single':
            # 单进程训练
            cmd = f"python train.py --num_games {args.games} --num_epochs {args.epochs} \
                  --num_simulations {args.simulations} --batch_size {args.batch_size} {gpu_flag} {gpu_id_flag}"
            print(f"启动单进程训练：{cmd}")
            os.system(cmd)
        else:
            # 多进程训练
            gpu_no_flag = "--no_gpu" if args.no_gpu else ""
            cmd = f"python multiproc_train.py --num_games {args.games} --num_epochs {args.epochs} \
                  --num_simulations {args.simulations} --batch_size {args.batch_size} {gpu_no_flag} {gpu_id_flag}"
            print(f"启动多进程训练：{cmd}")
            os.system(cmd)

    elif args.command == 'play':
        mcts_flag = "" if not args.no_mcts else "--no_mcts"
        gpu_flag = "" if args.no_gpu else "--device cuda"
        gpu_id_flag = f"--gpu_id {args.gpu_id}" if has_gpu and not args.no_gpu else ""
        cmd = f"python play.py --model {args.model} {mcts_flag} --simulations {args.simulations} {gpu_flag} {gpu_id_flag}"
        print(f"开始游戏：{cmd}")
        os.system(cmd)

    elif args.command == 'eval':
        # 创建临时评估脚本
        gpu_device = "cpu" if args.no_gpu else ("cuda" if has_gpu else "cpu")
        gpu_id = args.gpu_id if has_gpu and not args.no_gpu else 0
        
        script_content = f"""
import torch
import numpy as np
from tqdm import tqdm
from model import ReversiNet
from mcts import MCTS
from board import Board
from train import StateHistory

def evaluate_models(model1_path, model2_path, num_games, num_simulations, device="{gpu_device}", gpu_id={gpu_id}):
    # 设置设备
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and gpu_id is not None:
            device = f"cuda:{gpu_id}"
        print(f"使用 GPU: {{torch.cuda.get_device_name(0)}}")
    else:
        device = "cpu"
        print("使用 CPU")
    
    model1 = ReversiNet.load(model1_path, device=device)
    model2 = ReversiNet.load(model2_path, device=device)
    
    mcts1 = MCTS(model1, num_simulations)
    mcts2 = MCTS(model2, num_simulations)
    
    win1 = 0
    win2 = 0
    draw = 0
    
    for game in tqdm(range(num_games), desc=f"评估中"):
        board = Board()
        history = StateHistory()
        history.add(board.get_state())
        
        # 交替先手，确保公平
        if game % 2 == 0:
            model1_player = Board.BLACK  # 模型1先手
            model2_player = Board.WHITE  # 模型2后手
        else:
            model1_player = Board.WHITE  # 模型1后手
            model2_player = Board.BLACK  # 模型2先手
        
        player = Board.BLACK  # 游戏从黑棋开始
        
        while not board.is_game_over():
            legal_moves = board.get_legal_moves(player)
            
            if not legal_moves:
                player = -player
                continue
            
            # 根据当前玩家选择模型
            if player == model1_player:
                mcts = mcts1
            else:
                mcts = mcts2
            
            # 获取策略
            policy = mcts.run(board, player, history)
            
            # 选择最佳动作
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
            
            # 执行动作
            board.make_move(row, col, player)
            history.add(board.get_state())
            player = -player
        
        # 游戏结束，计算结果
        winner = board.get_winner()
        if winner is None:  # 平局
            draw += 1
        elif (winner == Board.BLACK and model1_player == Board.BLACK) or (winner == Board.WHITE and model1_player == Board.WHITE):
            win1 += 1  # 模型1赢
        else:
            win2 += 1  # 模型2赢
    
    # 计算胜率
    win_rate1 = (win1 + 0.5 * draw) / num_games
    win_rate2 = (win2 + 0.5 * draw) / num_games
    
    return win1, win2, draw, win_rate1, win_rate2

if __name__ == "__main__":
    win1, win2, draw, rate1, rate2 = evaluate_models(
        "{args.model1}", 
        "{args.model2}",
        {args.games},
        {args.simulations}
    )
    
    print(f"\\n评估结果 ({args.games} 局游戏):")
    print(f"模型1 ({args.model1}): 胜 {win1} 局, 负 {win2} 局, 平 {draw} 局, 胜率: {rate1:.2f}")
    print(f"模型2 ({args.model2}): 胜 {win2} 局, 负 {win1} 局, 平 {draw} 局, 胜率: {rate2:.2f}")
"""

        with open("temp_eval.py", "w") as f:
            f.write(script_content)

        print(f"评估模型: {args.model1} vs {args.model2}")
        os.system(f"python temp_eval.py")
        # 评估完成后删除临时脚本
        os.remove("temp_eval.py")


if __name__ == "__main__":
    main()
