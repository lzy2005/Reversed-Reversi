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

chessboard_size_ = 0
chessboard_ = None
color_ = 0
time_out_ = 0
start_time_ = 0

root = 0
state = []
parent = []
children = []
color = []
w = []
n = []
cnt = 0


def get_next_move():
	global state
	global parent
	global children
	global color
	global w
	global n
	global cnt

	state = [chessboard_]
	parent = [-1]
	children = [[]]
	color = [color_]
	w = [0]
	n = [0]
	cnt = 0

	while time.time() - start_time_ < time_out_ - 0.1:
		node = search()
		if n[node] == 0:
			backward(node, rollout(state[node].copy(), color[node]))
		else:
			if check_end(state[node]):
				backward(node, judge(state[node]))
				continue
			candidate_list = get_candidate_list(state[node], color[node])
			for pos in candidate_list:
				cnt += 1
				children[node].append(cnt)
				state.append(get_next_board(state[node].copy(), pos, color[node]))
				parent.append(node)
				children.append([])
				w.append(0)
				n.append(0)
				color.append(-color[node])
			if len(children[node]) == 0:
				cnt += 1
				children[node].append(cnt)
				state.append(state[node].copy())
				parent.append(node)
				children.append([])
				w.append(0)
				n.append(0)
				color.append(-color[node])
			node = uct_select(node)
			backward(node, rollout(state[node].copy(), color[node]))
	print(n[root])
	if color_ == COLOR_BLACK:  # Assume all children of root have been explored
		res = max(children[root], key=lambda i: w[i] / n[i])
	else:
		res = min(children[root], key=lambda i: w[i] / n[i])
	res = np.where((state[root] != 0) != (state[res] != 0))
	return (res[0][0], res[1][0])

def search():
	node = root
	while len(children[node]) != 0:
		node = uct_select(node)
	return node

def uct_select(node):
	log_total_n = math.log(n[node])
	if color[node] == COLOR_BLACK:
		return max(children[node], key=lambda i: w[i]/n[i]+EXPLORE_CON*math.sqrt(log_total_n/n[i]) if n[i]!=0 else math.inf)
	else:
		return max(children[node], key=lambda i: (n[i]-w[i])/n[i]+EXPLORE_CON*math.sqrt(log_total_n/n[i]) if n[i]!=0 else math.inf)

def backward(node, res):
	while node != -1:
		n[node]+=1
		w[node]+=res
		node = parent[node]

class AI(object):
	def __init__(self, chessboard_size, color, time_out):
		global chessboard_size_
		global color_
		global time_out_
		chessboard_size_ = chessboard_size
		color_ = color
		time_out_ = time_out

	def go( self, chessboard):
		global start_time_
		start_time_ = time.time()
		global chessboard_
		chessboard_ = chessboard
		candidate_list = get_candidate_list(chessboard_,color_)
		if len(candidate_list)!=0:
			candidate_list.append(get_next_move())
		return candidate_list

@numba.jit(nopython=True)
def checkin(x, y):
	return x >= 0 and x < chessboard_size_ and y >= 0 and y < chessboard_size_

@numba.jit(nopython=True)
def checkori(chessboard, x, y, color, i):
	nx = x + dx[i]
	ny = y + dy[i]
	if checkin(nx, ny) and chessboard[nx][ny] == -color:
		gx, gy = nx + dx[i], ny + dy[i]
		while checkin(gx, gy):
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