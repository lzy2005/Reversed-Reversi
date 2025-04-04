import numba
import numpy as np
import random
import time
import math

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
EXPLORE_CON=math.sqrt(2)

random.seed(0)

dx = np.array([0,1,1,1,0,-1,-1,-1], dtype=int)
dy = np.array([1,1,0,-1,-1,-1,0,1], dtype=int)

class MCTS(object):
	def __init__(self, ai, chessboard):
		self.chessboard_size_ = ai.chessboard_size
		self.chessboard_ = chessboard
		self.color_ = ai.color
		self.time_out_ = ai.time_out
		self.start_time_ = ai.start_time

		self.state = []
		self.parent = []
		self.children = []
		self.color = []
		self.w = []
		self.n = []
		self.root = 0
		self.cnt = 0

	@numba.jit(nopython=True)
	def get_next_move(self):
		self.state = [self.chessboard_]
		self.parent = [-1]
		self.children = [[]]
		self.color = [self.color_]
		self.w = [0]
		self.n = [0]
		self.cnt = 0

		while time.time() - self.start_time_ < self.time_out_ - 0.1:
			node = self.search()
			if self.n[node] == 0:
				self.backward(node, rollout(self.state[node].copy(), self.color[node]))
			else:
				if check_end(self.state[node]):
					self.backward(node, self.judge(self.state[node]))
					continue
				candidate_list = get_candidate_list(self.state[node], self.color[node])
				for pos in candidate_list:
					self.cnt += 1
					self.children[node].append(self.cnt)
					self.state.append(get_next_board(self.state[node].copy(), pos, self.color[node]))
					self.parent.append(node)
					self.children.append([])
					self.w.append(0)
					self.n.append(0)
					self.color.append(-self.color[node])
				if len(self.children[node]) == 0:
					self.cnt += 1
					self.children[node].append(self.cnt)
					self.state.append(self.state[node].copy())
					self.parent.append(node)
					self.children.append([])
					self.w.append(0)
					self.n.append(0)
					self.color.append(-self.color[node])
				node = self.uct_select(node)
				self.backward(node, rollout(self.state[node].copy(), self.color[node]))
		print(self.n[self.root])
		if self.color_ == COLOR_BLACK:  # Assume all children of root have been explored
			res = max(self.children[self.root], key=lambda i: self.w[i] / self.n[i])
		else:
			res = min(self.children[self.root], key=lambda i: self.w[i] / self.n[i])
		res = np.where((self.state[self.root] != 0) != (self.state[res] != 0))
		return (res[0][0], res[1][0])

	def search(self):
		node = self.root
		while len(self.children[node]) != 0:
			node = self.uct_select(node)
		return node

	def uct_select(self,node):
		log_total_n = math.log(self.n[node])
		if self.color[node] == COLOR_BLACK:
			return max(self.children[node], key=lambda i: self.w[i] / self.n[i] + EXPLORE_CON * math.sqrt(log_total_n / self.n[i]) if self.n[
																													  i] != 0 else math.inf)
		else:
			return max(self.children[node],
					   key=lambda i: (self.n[i] - self.w[i]) / self.n[i] + EXPLORE_CON * math.sqrt(log_total_n / self.n[i]) if self.n[
																											   i] != 0 else math.inf)

	def backward(self,node, res):
		while node != -1:
			self.n[node] += 1
			self.w[node] += res
			node = self.parent[node]

class AI(object):
	def __init__(self, chessboard_size, color , time_out):
		self.chessboard_size = chessboard_size
		self.color = color
		self.time_out = time_out
		self.candidate_list = []
		self.start_time = 0

	def go( self, chessboard):
		self.start_time = time.time()
		self.candidate_list = get_candidate_list(chessboard, self.color)
		if len(self.candidate_list)!=0:
			mcts = MCTS(self, chessboard)
			self.candidate_list.append(mcts.get_next_move())
		return self.candidate_list

@numba.jit(nopython=True)
def checkin(x, y, size):
	return x >= 0 and x < size and y >= 0 and y < size

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def judge(chessboard):
	sum_black = (chessboard==-1).sum()
	sum_white = (chessboard== 1).sum()
	if sum_black <sum_white:
		return 1
	if sum_black  == sum_white:
		return 0.5
	return 0

@numba.jit(nopython=True)
def get_next_board(chessboard, pos, color):
	chessboard[pos[0]][pos[1]]=color
	for i in range(8):
		if checkori(chessboard, pos[0], pos[1], color, i):
			nxt = (pos[0] + dx[i], pos[1] + dy[i])
			while chessboard[nxt[0]][nxt[1]] == -color:
				chessboard[nxt[0]][nxt[1]] = color
				nxt = (nxt[0] + dx[i], nxt[1] + dy[i])
	return chessboard

@numba.jit(nopython=True)
def check_end(chessboard):
	if len(get_candidate_list(chessboard, COLOR_BLACK)) > 0:
		return False
	if len(get_candidate_list(chessboard, COLOR_WHITE)) > 0:
		return False
	return True

@numba.jit(nopython=True)
def rollout(chessboard, color):
	while check_end(chessboard) == False:
		candidate_list = get_candidate_list(chessboard, color)
		if len(candidate_list) == 0:
			color = -color
		else:
			get_next_board(chessboard, candidate_list[random.randint(0, len(candidate_list) - 1)], color)
			color = -color
	return judge(chessboard)