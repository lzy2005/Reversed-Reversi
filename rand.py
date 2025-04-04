from operator import truediv

import numpy as np
import random
import time

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
random.seed(0)

dx = [0,1,1,1,0,-1,-1,-1]
dy = [1,1,0,-1,-1,-1,0,1]

class AI(object):
	def __init__(self, chessboard_size, color , time_out):
		self.chessboard_size = chessboard_size
		self.color = color
		self.time_out = time_out
		self.candidate_list = []
	def	checkin(self, nx, ny):
		return nx>=0 and nx<8 and ny>=0 and ny<8
	def checkori(self, chessboard, x, y, i):
		nx, ny = x+dx[i],y+dy[i]
		if self.checkin(nx,ny) and chessboard[nx][ny]==-self.color:
			gx, gy = nx+dx[i], ny+dy[i]
			while self.checkin(gx, gy):
				if chessboard[gx][gy]==0:
					return True
				if chessboard[gx][gy]==self.color:
					return False
				gx, gy = gx+dx[i], gy+dy[i]
			return False
		else:
			return False
	def check(self, chessboard, x, y):
		for i in range(8):
			if self.checkori(chessboard,x,y,i):
				return True
		return False
	def go( self, chessboard):
		self.candidate_list.clear()
		idx = np.where(chessboard == COLOR_NONE)
		idx = list(zip(idx[0], idx[1]))
		for (x,y) in idx:
			if self.check(chessboard, x, y):
				self.candidate_list.append((x,y))
		if len(self.candidate_list)!=0:
			self.candidate_list.append(self.candidate_list[random.randint(0,len(self.candidate_list)-1)])
		return self.candidate_list

# start = time.time()
# run_time = (time.time() - start)