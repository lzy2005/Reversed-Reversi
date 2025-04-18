import numba
import numpy as np
import random
import time
import math

EXPLORE_CON = math.sqrt(2)
OPT_CON = 0.5

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0

random.seed(0)

dx = np.array([0,1,1,1,0,-1,-1,-1], dtype=int)
dy = np.array([1,1,0,-1,-1,-1,0,1], dtype=int)

class Node(object):
	def __init__(self, state, parent, color):
		self.state = state
		self.parent = parent
		self.children = []
		self.color = color
		self.w = 0
		self.n = 0

class MCTS(object):
	def __init__(self, ai, chessboard):
		self.chessboard_size = ai.chessboard_size
		self.chessboard = chessboard
		self.color = ai.color
		self.time_out = ai.time_out
		self.start_time = ai.start_time
		self.root = Node(self.chessboard, None, self.color)
		self.num_of_nodes = 1

	def print_situation(self, res):
		print(f"total N: {self.root.n}")
		print(f"pio win rate: {res.w/res.n}")
		print(f"current color: {self.color}")
		print(f"situation of all steps: {[(c.w/c.n, c.n) for c in self.root.children]}")
		print(f"num of nodes: {self.num_of_nodes}")

	def get_next_move(self):
		while time.time()-self.start_time < self.time_out-0.01:
			node = self.search()
			if node.n==0:
				self.backward(node, rollout(node.state.copy(),node.color))
			else:
				if check_end(node.state):
					self.backward(node, judge(node.state))
					continue
				candidate_list = get_candidate_list(node.state,node.color)
				for pos in candidate_list:
					node.children.append(Node(get_next_board(node.state.copy(),pos,node.color), node, -node.color))
					self.num_of_nodes += 1
				if len(node.children) == 0:
					node.children.append(Node(node.state.copy(), node, -node.color))
					self.num_of_nodes += 1
				node = self.uct_select(node)
				self.backward(node, rollout(node.state.copy(),node.color))
		if self.color == COLOR_BLACK: # Assume all children of root have been explored
			res = max(self.root.children, key= lambda n: n.w/n.n)
		else:
			res = min(self.root.children, key=lambda n: n.w/n.n)
		self.print_situation(res)
		res = np.where((self.root.state!=0)!=(res.state!=0))
		return (res[0][0],res[1][0])

	def search(self):
		node = self.root
		while len(node.children) != 0:
			node = self.uct_select(node)
		return node

	def uct_select(self, node):
		log_total_n=math.log(node.n)
		if node.color == COLOR_BLACK:
			return max(node.children, key=lambda n: n.w/n.n+EXPLORE_CON*math.sqrt(log_total_n/n.n) if n.n!=0 else math.inf)
		else:
			return max(node.children, key=lambda n: (n.n-n.w)/n.n+EXPLORE_CON*math.sqrt(log_total_n/n.n) if n.n!=0 else math.inf)

	def backward(self, node, w):
		while node != None:
			node.n+=1
			node.w+=w
			node = node.parent

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
	if sum_black < sum_white:
		return 1 + (sum_white - sum_black) / 64 * OPT_CON
	if sum_black == sum_white:
		return 0.5
	return 0 + (sum_white - sum_black) / 64 * OPT_CON

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