import numpy as np
import numba
from numba.typed import List
import random
import time
import math

SIZE = 500000
SIZE_U = SIZE + 100

GAMA = 0.95
TIME_END = 0.03 #

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0

CHESSBOARD_SIZE = 8
GRID_NUM = CHESSBOARD_SIZE ** 2

random.seed(0)

dx = np.array([0,1,1,1,0,-1,-1,-1], dtype=int)
dy = np.array([1,1,0,-1,-1,-1,0,1], dtype=int)

@numba.jit(numba.boolean(
	numba.int8,
	numba.int8,
numba.int8), nopython=True)
def checkin(x, y, size):
	return x >= 0 and x < size and y >= 0 and y < size

@numba.jit(numba.boolean(
	numba.types.Array(numba.int8, 2, 'C'),
	numba.int8,
	numba.int8,
	numba.int8,
numba.int8), nopython=True)
def checkori(chessboard, x, y, color, i):
	size = chessboard.shape[0]
	nx = x + dx[i]
	ny = y + dy[i]
	if checkin(nx, ny, size) and chessboard[nx][ny] == -color:
		gx, gy = nx + dx[i], ny + dy[i]
		while checkin(gx, gy, size):
			if chessboard[gx][gy] == color:
				return True
			if chessboard[gx][gy] == 0:
				return False
			gx, gy = gx + dx[i], gy + dy[i]
		return False
	else:
		return False

@numba.jit(numba.boolean(
	numba.types.Array(numba.int8, 2, 'C'),
	numba.int8,
	numba.int8,
numba.int8), nopython=True)
def check(chessboard, x, y, color):
	for i in range(8):
		if checkori(chessboard, x, y, color, i):
			return True
	return False

@numba.jit(numba.types.ListType(numba.types.Tuple([numba.types.int64, numba.types.int64]))(
    numba.types.Array(numba.types.int8, 2, 'C'), numba.types.int8
), nopython=True)
def get_candidate_list(chessboard, color):
	idx = np.where(chessboard == COLOR_NONE)
	candidate_list = List()
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]
		if check(chessboard, x, y, color):
			candidate_list.append((x,y))
	return candidate_list

@numba.jit(numba.types.Array(numba.int8, 2, 'C')(
	numba.types.Array(numba.int8, 2, 'C'),
	numba.types.Tuple((numba.int32, numba.int32)),
numba.int8), nopython=True)
def get_next_board(chessboard, pos, color):
	chessboard[pos[0], pos[1]]=color
	for i in range(8):
		if checkori(chessboard, pos[0], pos[1], color, i):
			nxt = (pos[0] + dx[i], pos[1] + dy[i])
			while chessboard[nxt[0]][nxt[1]] == -color:
				chessboard[nxt[0], nxt[1]] = color
				nxt = (nxt[0] + dx[i], nxt[1] + dy[i])
	return chessboard

@numba.jit(numba.boolean(numba.types.Array(numba.int8, 2, 'C')), nopython=True)
def check_end(chessboard):
	if len(get_candidate_list(chessboard, COLOR_BLACK)) > 0:
		return False
	if len(get_candidate_list(chessboard, COLOR_WHITE)) > 0:
		return False
	return True

@numba.jit(numba.types.float32(
	numba.types.Array(numba.int8, 2, 'C')
), nopython=True)
def judge(chessboard):
	sum_black = (chessboard == -1).sum()
	sum_white = (chessboard == 1).sum()
	if sum_black < sum_white:
		return 1e8
	if sum_black == sum_white:
		return 0
	return -1e8

@numba.jit(numba.types.float32(
	numba.types.Array(numba.int8, 2, 'C')
), nopython=True)
def eval(chessboard):
	sum_black = (chessboard == -1).sum()
	sum_white = (chessboard == 1).sum()
	depth = sum_black + sum_white
	con = GAMA ** (GRID_NUM - depth)

	sum_val = sum_white - sum_black
	corner_val = (chessboard[0, 0] + chessboard[0, -1] + chessboard[-1, 0] + chessboard[-1, -1]) * 16

	return con * sum_val + (1-con) * corner_val

@numba.jit(numba.types.Array(numba.int8, 2, 'C')(
	numba.types.Array(numba.int8, 2, 'C'),
	numba.types.Tuple((numba.int32, numba.int32)),
numba.int8), nopython=True)
def copy_get_next_board(chessboard, pos, color):
	return get_next_board(chessboard.copy(), pos, color)

def adv_search(chessboard, color, alpha, beta, depth, max_depth, end_time):
	if check_end(chessboard):
		return judge(chessboard), (0, 0)
	if depth == max_depth:
		return eval(chessboard), (0, 0)
	if end_time - time.time() < TIME_END:
		return 0, (-1, -1)

	candidate_list = get_candidate_list(chessboard, color)
	if len(candidate_list) == 0:
		return adv_search(chessboard, -color, alpha, beta, depth, max_depth, end_time)[0], None
	if color == COLOR_BLACK:
		value, strategy = -math.inf, None
		for pos in candidate_list:
			n_value, n_strategy = adv_search(copy_get_next_board(chessboard, pos, color), -color, alpha, beta, depth+1, max_depth, end_time)
			if n_strategy == (-1, -1):
				return 0, (-1, -1)
			if n_value > value:
				value, strategy = n_value, pos
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return value, strategy
	else:
		value, strategy = math.inf, None
		for pos in candidate_list:
			n_value, n_strategy = adv_search(copy_get_next_board(chessboard, pos, color), -color, alpha, beta, depth+1, max_depth, end_time)
			if n_strategy == (-1, -1):
				return 0, (-1, -1)
			if n_value < value:
				value, strategy = n_value, pos
			beta = min(beta, value)
			if beta <= alpha:
				break
		return value, strategy

def get_next_move(chessboard, color, end_time):
	candidate_list = get_candidate_list(chessboard, color)
	val, res = 0, candidate_list[0]
	for max_depth in range(1, GRID_NUM - ((chessboard != 0).sum()) + 1):
		if end_time - time.time() < TIME_END:
			break
		n_val, n_res = adv_search(chessboard, color, -math.inf, math.inf, 0, max_depth, end_time)
		if n_res != (-1, -1):
			val, res = n_val, n_res
		print(f"now depth {max_depth} | now time {time.time() - end_time + 3} | now val {val}")
	return res

class AI(object):
	def __init__(self, chessboard_size, color , time_out):
		self.start_time = time.time()
		self.chessboard_size = chessboard_size
		self.color = color
		self.time_out = time_out
		self.candidate_list = None

	def go(self, chessboard):
		self.start_time = time.time()
		chessboard = chessboard.astype(np.int8)
		self.candidate_list = list(get_candidate_list(chessboard, self.color))
		if len(self.candidate_list)!=0:
			self.candidate_list.append(get_next_move(chessboard, self.color, self.start_time + self.time_out))
		return self.candidate_list