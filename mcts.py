import numpy as np
import torch
from typing import List, Tuple, Optional
from board import Board
from model import ReversiNet

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.virtual_loss = 0
        
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return (self.value_sum - self.virtual_loss) / self.visit_count

class MCTS:
    def __init__(self, model: ReversiNet, num_simulations: int = 800, c_puct: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.virtual_loss = 1.0  # 虚拟损失值
        
    def run(self, board: Board, player: int, history=None) -> np.ndarray:
        """
        运行MCTS搜索
        Args:
            board: 当前棋盘状态
            player: 当前玩家
            history: 可选的历史状态记录器
        Returns:
            落子概率分布
        """
        root = Node(0)
        root.state = board.get_state()
        
        # 如果没有提供历史，创建一个新的历史对象
        if history is None:
            from train import StateHistory
            history = StateHistory()
            history.add(root.state)
            
        # 并行模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_board = board.copy()
            current_player = player
            
            # 为搜索创建历史的副本
            search_history = type(history)()
            for state in history.history:
                search_history.add(state)
            
            # 选择
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                row, col = action // 8, action % 8
                current_board.make_move(row, col, current_player)
                search_history.add(current_board.get_state())
                current_player = -current_player
                
                # 应用虚拟损失
                node.virtual_loss += self.virtual_loss
            
            # 扩展和评估
            if not current_board.is_game_over():
                state_tensor = self._prepare_state(search_history)
                policy, value = self.model.predict(state_tensor)
                
                # 创建子节点
                legal_moves = current_board.get_legal_moves(current_player)
                for move in legal_moves:
                    move_idx = move[0] * 8 + move[1]
                    node.children[move_idx] = Node(policy[0, move_idx].item())
                
                # 反向传播
                self.backpropagate(search_path, value.item(), player)
            else:
                # 游戏结束，使用实际结果
                winner = current_board.get_winner()
                if winner is not None:
                    value = 1 if winner == player else -1
                else:
                    value = 0
                self.backpropagate(search_path, value, player)
            
            # 移除虚拟损失
            for node in search_path[1:]:
                node.virtual_loss -= self.virtual_loss
        
        # 计算落子概率
        visit_counts = np.zeros(64)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            
        if np.sum(visit_counts) > 0:
            policy = visit_counts / np.sum(visit_counts)
        else:
            policy = np.zeros(64)  # 8x8棋盘
            
        return policy
    
    def select_child(self, node: Node) -> Tuple[int, Node]:
        """
        选择UCB值最大的子节点
        """
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            # 计算UCB值
            ucb = child.value() + self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def backpropagate(self, search_path: List[Node], value: float, player: int):
        """
        反向传播更新节点统计信息
        """
        for node in reversed(search_path):
            node.value_sum += value if player == 1 else -value
            node.visit_count += 1
            player = -player
    
    def _prepare_state(self, history) -> torch.Tensor:
        """
        将带历史的棋盘状态转换为模型输入格式
        """
        state_tensor = torch.tensor(history.get_state_with_history(), dtype=torch.float32)
        return state_tensor.unsqueeze(0) 