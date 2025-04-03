import numpy as np
from typing import List, Tuple, Optional

class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    
    def __init__(self, size: int = 8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self._init_board()
        
    def _init_board(self):
        """初始化棋盘,在中间放置初始棋子"""
        mid = self.size // 2
        self.board[mid-1][mid-1] = self.WHITE
        self.board[mid-1][mid] = self.BLACK
        self.board[mid][mid-1] = self.BLACK
        self.board[mid][mid] = self.WHITE
        
    def get_legal_moves(self, player: int) -> List[Tuple[int, int]]:
        """获取当前玩家的合法落子位置"""
        legal_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == self.EMPTY:
                    if self._is_valid_move(i, j, player):
                        legal_moves.append((i, j))
        return legal_moves
    
    def _is_valid_move(self, row: int, col: int, player: int) -> bool:
        """检查在指定位置落子是否合法"""
        if self.board[row][col] != self.EMPTY:
            return False
            
        # 检查8个方向
        directions = [(0,1), (1,0), (0,-1), (-1,0),
                     (1,1), (-1,-1), (1,-1), (-1,1)]
        
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, player):
                return True
        return False
    
    def _check_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> bool:
        """检查某个方向是否可以形成有效落子"""
        r, c = row + dr, col + dc
        opponent = -player
        has_opponent = False
        
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == opponent:
                has_opponent = True
            elif self.board[r][c] == player and has_opponent:
                return True
            else:
                break
            r += dr
            c += dc
        return False
    
    def make_move(self, row: int, col: int, player: int) -> bool:
        """执行落子操作"""
        if not self._is_valid_move(row, col, player):
            return False
            
        self.board[row][col] = player
        directions = [(0,1), (1,0), (0,-1), (-1,0),
                     (1,1), (-1,-1), (1,-1), (-1,1)]
        
        for dr, dc in directions:
            self._flip_direction(row, col, dr, dc, player)
        return True
    
    def _flip_direction(self, row: int, col: int, dr: int, dc: int, player: int):
        """翻转某个方向的棋子"""
        r, c = row + dr, col + dc
        opponent = -player
        to_flip = []
        
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r][c] == opponent:
                to_flip.append((r, c))
            elif self.board[r][c] == player:
                for flip_r, flip_c in to_flip:
                    self.board[flip_r][flip_c] = player
                break
            else:
                break
            r += dr
            c += dc
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        # 检查是否有空位
        if np.any(self.board == self.EMPTY):
            # 检查双方是否还有合法落子
            return len(self.get_legal_moves(self.BLACK)) == 0 and len(self.get_legal_moves(self.WHITE)) == 0
        return True
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        if not self.is_game_over():
            return None
            
        black_count = np.sum(self.board == self.BLACK)
        white_count = np.sum(self.board == self.WHITE)
        
        if black_count > white_count:
            return self.BLACK
        elif white_count > black_count:
            return self.WHITE
        return None
    
    def get_state(self) -> np.ndarray:
        """获取当前棋盘状态"""
        return self.board.copy()
    
    def __str__(self) -> str:
        """打印棋盘状态"""
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        return '\n'.join([' '.join(symbols[cell] for cell in row) for row in self.board]) 