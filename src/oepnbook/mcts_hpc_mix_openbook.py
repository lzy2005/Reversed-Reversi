import numpy as np
import numba
import random
import time
import math
from numba.experimental import jitclass

SIZE = 1000000
SIZE_U = SIZE + 100

EXPLORE_CON = math.sqrt(2)
OPT_CON = 0.5

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0

random.seed(0)

dx = np.array([0,1,1,1,0,-1,-1,-1], dtype=int)
dy = np.array([1,1,0,-1,-1,-1,0,1], dtype=int)

spec = [
    ('chessboard_size', numba.int8),
    ('chessboard', numba.types.Array(numba.int8, 2, 'C')),
    ('origin_color', numba.int8),
    ('time_out', numba.float32),
    ('start_time', numba.float64),
    ('state', numba.types.Array(numba.int8, 3, 'C')),
    ('parent', numba.types.Array(numba.int32, 1, 'C')),
    ('head', numba.types.Array(numba.int32, 1, 'C')),
    ('next', numba.types.Array(numba.int32, 1, 'C')),
    ('color', numba.types.Array(numba.int8, 1, 'C')),
    ('w', numba.types.Array(numba.float32, 1, 'C')),
    ('n', numba.types.Array(numba.int32, 1, 'C')),
    ('last', numba.int32),
    ('root', numba.int32),
    ('children', numba.types.Array(numba.int32, 1, 'C')),
    ('vals', numba.types.Array(numba.float32, 1, 'C')),
    ('children_cnt', numba.int32)
]

@numba.jit(numba.boolean(numba.int8, numba.int8, numba.int8), nopython=True)
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

@numba.jit(nopython=True)
def get_candidate_list(chessboard, color):
	idx = np.where(chessboard == COLOR_NONE)
	candidate_list = []
	for i in range(len(idx[0])):
		x, y = idx[0][i], idx[1][i]
		if check(chessboard, x, y, color):
			candidate_list.append((x,y))
	return candidate_list

@numba.jit(numba.types.float32(
	numba.types.Array(numba.int8, 2, 'C')
), nopython=True)
def judge(chessboard):
	sum_black = (chessboard == -1).sum()
	sum_white = (chessboard == 1).sum()
	if sum_black < sum_white:
		return 1 + (sum_white - sum_black) / 64 * OPT_CON
	if sum_black == sum_white:
		return 0.5
	return 0 + (sum_white - sum_black) / 64 * OPT_CON

@numba.jit(numba.types.Array(numba.int8, 2, 'C')(
	numba.types.Array(numba.int8, 2, 'C'),
	numba.types.Tuple((numba.int32, numba.int32)),
numba.int8), nopython=True)
def get_next_board(chessboard, pos, color):
	chessboard[pos[0]][pos[1]]=color
	for i in range(8):
		if checkori(chessboard, pos[0], pos[1], color, i):
			nxt = (pos[0] + dx[i], pos[1] + dy[i])
			while chessboard[nxt[0]][nxt[1]] == -color:
				chessboard[nxt[0]][nxt[1]] = color
				nxt = (nxt[0] + dx[i], nxt[1] + dy[i])
	return chessboard

@numba.jit(numba.boolean(numba.types.Array(numba.int8, 2, 'C')), nopython=True)
def check_end(chessboard):
	if len(get_candidate_list(chessboard, COLOR_BLACK)) > 0:
		return False
	if len(get_candidate_list(chessboard, COLOR_WHITE)) > 0:
		return False
	return True

@jitclass(spec)
class MCT(object):
	def __init__(self, chessboard_size, color, time_out, start_time, chessboard):
		self.chessboard_size = chessboard_size
		self.chessboard = chessboard
		self.origin_color = color
		self.time_out = time_out
		self.start_time = start_time

		self.state = np.zeros((SIZE_U, self.chessboard_size, self.chessboard_size), dtype=np.int8)
		self.parent = np.zeros(SIZE_U, dtype=np.int32)
		self.head = np.zeros(SIZE_U, dtype=np.int32)
		self.next = np.zeros(SIZE_U, dtype=np.int32)
		self.color = np.zeros(SIZE_U, dtype=np.int8)
		self.w = np.zeros(SIZE_U, dtype=np.float32)
		self.n = np.zeros(SIZE_U, dtype=np.int32)
		self.last = -1

		self.root = -1

		self.children = np.zeros(100, dtype=np.int32)
		self.vals = np.zeros(100, dtype=np.float32)
		self.children_cnt = 0

@numba.jit(nopython=True)
def add_node(mct, state, parent, color):
	mct.last += 1
	mct.next[mct.last] = mct.head[parent]
	mct.head[parent] = mct.last
	mct.state[mct.last] = state
	mct.parent[mct.last] = parent
	mct.head[mct.last] = -1
	mct.color[mct.last] = color
	mct.w[mct.last] = 0
	mct.n[mct.last] = 0
	return mct.last

def print_situation(mct, res):
	print(f"total N: {mct.n[mct.root]}")
	print(f"pio win rate: {mct.w[res]/mct.n[res]}")
	print(f"current color: {mct.color[mct.root]}")
	son = mct.head[mct.root]
	print("situation of all sons:")
	while son != -1:
		print(f"({mct.w[son]/mct.n[son]}, {mct.n[son]})", end=' ')
		son = mct.next[son]
	print()
	print(f"num of nodes: {mct.last + 1}")

@numba.jit(nopython=True)
def aug(mct):
	node = search(mct)
	if mct.n[node] == 0:
		backward(mct, node, rollout(mct.state[node].copy(), mct.color[node]))
	else:
		if check_end(mct.state[node]):
			backward(mct, node, judge(mct.state[node]))
			return
		if mct.last > SIZE:
			backward(mct, node, rollout(mct.state[node].copy(), mct.color[node]))
			return
		candidate_list = get_candidate_list(mct.state[node], mct.color[node])
		for pos in candidate_list:
			add_node(mct, get_next_board(mct.state[node].copy(), pos, mct.color[node]), node, -mct.color[node])
		if mct.head[node] == -1:
			add_node(mct, mct.state[node], node, -mct.color[node])
		node = uct_select(mct, node)
		backward(mct, node, rollout(mct.state[node].copy(), mct.color[node]))

@numba.jit(nopython=True)
def get_final_move(mct):
	mct.children_cnt, child = 0, mct.head[mct.root]
	while child != -1:
		mct.children[mct.children_cnt] = child
		mct.vals[mct.children_cnt] = mct.w[child]/mct.n[child]
		mct.children_cnt += 1
		child = mct.next[child]
	if mct.color[mct.root] == COLOR_BLACK:
		res=mct.children[np.argmax(mct.vals[0: mct.children_cnt])]
	else:
		res=mct.children[np.argmin(mct.vals[0: mct.children_cnt])]
	# print_situation(mct, res) #
	res = np.where((mct.state[mct.root]!=0)!=(mct.state[res]!=0))
	return (res[0][0],res[1][0])

def get_next_move(mct):
	while time.time()-mct.start_time < mct.time_out-0.1:
		aug(mct)

	return get_final_move(mct)

@numba.jit(nopython=True)
def search(mct):
	node = mct.root
	while mct.head[node] != -1:
		node = uct_select(mct, node)
	return node

@numba.jit(nopython=True)
def uct_select(mct, node):
	log_total_n=math.log(mct.n[node])
	mct.children_cnt, child = 0, mct.head[node]
	while child != -1:
		mct.children[mct.children_cnt] = child
		if mct.color[node] == COLOR_BLACK:
			mct.vals[mct.children_cnt] = mct.w[child] / mct.n[child] + EXPLORE_CON * math.sqrt(
				log_total_n / mct.n[child]) if mct.n[child] != 0 else math.inf
		else:
			mct.vals[mct.children_cnt] = (mct.n[child] - mct.w[child]) / mct.n[child] + EXPLORE_CON * math.sqrt(
				log_total_n / mct.n[child]) if mct.n[child] != 0 else math.inf
		mct.children_cnt += 1
		child = mct.next[child]
	return mct.children[np.argmax(mct.vals[0: mct.children_cnt])]

@numba.jit(nopython=True)
def backward(mct, node, w):
	while node != -1:
		mct.n[node]+=1
		mct.w[node]+=w
		node = mct.parent[node]

@numba.jit(numba.float32(
	numba.types.Array(numba.int8, 2, 'C'),
numba.int8), nopython=True)
def rollout(chessboard, color):
	while check_end(chessboard) == False:
		candidate_list = get_candidate_list(chessboard, color)
		if len(candidate_list) == 0:
			color = -color
		else:
			get_next_board(chessboard, candidate_list[random.randint(0, len(candidate_list) - 1)], color)
			color = -color
	return judge(chessboard)

class AI(object):
	def __init__(self, chessboard_size, color , time_out):
		self.chessboard_size = chessboard_size
		self.color = color
		self.time_out = time_out
		self.candidate_list = []
		self.start_time = 0

	def go(self, chessboard):
		self.start_time = time.time()
		chessboard = chessboard.astype(np.int8)
		self.candidate_list = get_candidate_list(chessboard, self.color)
		if len(self.candidate_list)!=0:
			mct = MCT(self.chessboard_size, self.color, self.time_out, self.start_time, chessboard)
			mct.root = add_node(mct, chessboard, -1, self.color)
			self.candidate_list.append(get_next_move(mct))
		return self.candidate_list