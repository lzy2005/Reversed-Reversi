import numpy as np
import numba
from numba.typed import List
from numba.experimental import jitclass
import time
import math

spec = [
    ('state', numba.types.Array(numba.int8, 3, 'C')),
    ('color', numba.types.Array(numba.int8, 1, 'C')),
    ('next', numba.types.Array(numba.int8, 3, 'C')),
	('next_len', numba.types.Array(numba.int8, 1, 'C')),
	("depth", numba.types.Array(numba.int8, 1, 'C')),
	("alpha", numba.types.Array(numba.float32, 1, 'C')),
	("beta", numba.types.Array(numba.float32, 1, 'C')),
	("value", numba.types.Array(numba.float32, 1, 'C')),
	("pos", numba.types.int8),
	("res", numba.types.Tuple([numba.int64, numba.int64])),
	("ended", numba.boolean)
]

@jitclass(spec)
class AST():
	def __init__(self):
		self.state = np.zeros((64 * 2, 8, 8), dtype=np.int8)
		self.color = np.zeros(64 * 2, dtype=np.int8)
		self.next = np.zeros((64 * 2, 64, 2), dtype=np.int8)
		self.next_len = np.zeros(64 * 2, dtype=np.int8)
		self.depth = np.zeros(64 * 2, dtype=np.int8)
		self.alpha = np.zeros(64 * 2, dtype=np.float32)
		self.beta = np.zeros(64 * 2, dtype=np.float32)
		self.value = np.zeros(64 * 2, dtype=np.float32)
		self.pos = 0
		self.res = (-1, -1)
		self.ended = True

AST_TYPE = AST.class_type.instance_type

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
	idx = np.where(chessboard == 0)
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
	idx = np.where(chessboard == 0)
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]
		if check(chessboard, x, y, -1) or check(chessboard, x, y, 1):
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

@numba.jit(numba.void(
	numba.int8, numba.int8, numba.int8, numba.int8, numba.boolean, numba.types.Array(numba.int8, 1, 'C')
), nopython=True)
def add_edge_val(o, a, b, c, con, edge_val):
	if o != 0:
		return
	if a == b and b == c and c != 0 and con:
		edge_val[0] += -a
	else:
		edge_val[1] += -(a + b)

@numba.jit(numba.types.float32(
	numba.types.Array(numba.int8, 2, 'C')
), nopython=True)
def eval(chessboard):
	sum_black =  (chessboard == -1).sum()
	sum_white = (chessboard == 1).sum()
	depth = 64 - (sum_black + sum_white)

	sum_val = sum_white - sum_black
	corner_val = chessboard[0, 0] + chessboard[0, -1] + chessboard[-1, 0] + chessboard[-1, -1]

	edge_val = np.array([0, 0], dtype=np.int8)
	add_edge_val(chessboard[0, 0], chessboard[0, 1], chessboard[1, 0], chessboard[1, 1],chessboard[1, 1] == chessboard[0, 2] or chessboard[1, 1] == chessboard[2, 0], edge_val)
	add_edge_val(chessboard[0, 7], chessboard[0, 6], chessboard[1, 7], chessboard[1, 6],chessboard[1, 6] == chessboard[0, 5] or chessboard[1, 6] == chessboard[2, 7], edge_val)
	add_edge_val(chessboard[7, 7], chessboard[7, 6], chessboard[6, 7], chessboard[6, 6],chessboard[6, 6] == chessboard[5, 7] or chessboard[6, 6] == chessboard[7, 5], edge_val)
	add_edge_val(chessboard[7, 0], chessboard[6, 0], chessboard[7, 1], chessboard[6, 1],chessboard[6, 1] == chessboard[5, 0] or chessboard[6, 1] == chessboard[7, 2], edge_val)

	if depth >= 16:
		return corner_val * 25 + edge_val[0] * 12 + edge_val[1] * 4 + sum_val
	elif depth > 4:
		return corner_val * 16 + edge_val[0] * 10 + edge_val[1] * 3 + sum_val
	else:
		return corner_val * 6 + edge_val[0] * 8 + edge_val[1] * 2 + sum_val
	# spec +

@numba.jit(numba.void(AST_TYPE), nopython=True)
def clear(ast):
	ast.pos = -1
	ast.res = (-1, -1)
	ast.ended = False

@numba.jit(numba.void(
	AST_TYPE,
	numba.types.ListType(numba.types.Tuple([numba.types.int64, numba.types.int64]))
), nopython=True)
def insert_next(ast, candidate_list):
	ast.next_len[ast.pos] = len(candidate_list)
	for i in range(ast.next_len[ast.pos]):
		ast.next[ast.pos][i][0] = candidate_list[i][0]
		ast.next[ast.pos][i][1] = candidate_list[i][1]
	if len(candidate_list) == 0:
		ast.next_len[ast.pos] = 1
		ast.next[ast.pos][0][0] = -1
		ast.next[ast.pos][0][1] = -1

@numba.jit(numba.void(
	AST_TYPE,
	numba.types.Array(numba.int8, 2, 'C'),
	numba.int8,
	numba.int8,
	numba.int8
), nopython=True)
def extend(ast, chessboard, color, depth, max_depth):
	ast.pos += 1

	if check_end(chessboard):
		ast.value[ast.pos] = judge(chessboard)
		ast.next_len[ast.pos] = 0
		return
	if depth == max_depth:
		ast.value[ast.pos] = eval(chessboard)
		ast.next_len[ast.pos] = 0
		return

	ast.state[ast.pos] = chessboard
	ast.color[ast.pos] = color
	insert_next(ast, get_candidate_list(chessboard, color))
	ast.depth[ast.pos] = depth
	if ast.pos == 0:
		ast.alpha[ast.pos] = -math.inf
		ast.beta[ast.pos] = math.inf
	else:
		ast.alpha[ast.pos] = ast.alpha[ast.pos - 1]
		ast.beta[ast.pos] = ast.beta[ast.pos - 1]
	if color == -1:
		ast.value[ast.pos] = -math.inf
	else:
		ast.value[ast.pos] = math.inf

@numba.jit(numba.void(
	AST_TYPE,
	numba.int8
), nopython=True)
def adv_iterate(ast, max_depth):
	for i in range(1000):
		while ast.next_len[ast.pos] == 0 or ast.alpha[ast.pos] >= ast.beta[ast.pos]:
			ast.pos -= 1
			if ast.pos == -1:
				ast.ended = True
				break
			if ast.color[ast.pos] == -1:
				if ast.value[ast.pos + 1] > ast.value[ast.pos]:
					ast.value[ast.pos] = ast.value[ast.pos + 1]
					ast.alpha[ast.pos] = max(ast.alpha[ast.pos], ast.value[ast.pos])
					if ast.pos == 0:
						ast.res = (ast.next[ast.pos][ast.next_len[ast.pos]][0], ast.next[ast.pos][ast.next_len[ast.pos]][1])
			else:
				if ast.value[ast.pos + 1] < ast.value[ast.pos]:
					ast.value[ast.pos] = ast.value[ast.pos + 1]
					ast.beta [ast.pos] = min(ast.beta [ast.pos], ast.value[ast.pos])
					if ast.pos == 0:
						ast.res = (ast.next[ast.pos][ast.next_len[ast.pos]][0], ast.next[ast.pos][ast.next_len[ast.pos]][1])
		if ast.ended:
			break

		ast.next_len[ast.pos] -= 1
		pos = (ast.next[ast.pos][ast.next_len[ast.pos]][0], ast.next[ast.pos][ast.next_len[ast.pos]][1])
		if pos != (-1, -1):
			extend(ast, get_next_board(ast.state[ast.pos].copy(), pos, ast.color[ast.pos]), -ast.color[ast.pos], ast.depth[ast.pos] + 1 , max_depth)
		else:
			extend(ast, ast.state[ast.pos], -ast.color[ast.pos], ast.depth[ast.pos], max_depth)

def get_next_move(chessboard, color, end_time):
	ast = AST()
	candidate_list = get_candidate_list(chessboard, color)
	res = candidate_list[0]
	sum_chess = (chessboard != 0).sum()
	for max_depth in range(1, 64 - sum_chess + 1):
		if end_time - time.time() < 0.03:
			break
		clear(ast)
		extend(ast, chessboard, color, 0, max_depth)
		while ast.ended == False and end_time -time.time() >= 0.03:
			adv_iterate(ast, max_depth)
		if ast.ended ==True:
			res = ast.res
		print(f"now depth {max_depth} | now time {time.time() - end_time + 3} | now val {ast.value[0]} | hpc")
	print(f" ext time {time.time() - end_time + 0.03}")
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