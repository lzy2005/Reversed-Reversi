from hpc import mcts_hpc_mix_complie0 as model1 # pio
from baseline import mcts_baseline as model2
import numpy as np

CHESSBOARD_SIZE = 8
TIME_OUT = 3

player = [model1.AI(CHESSBOARD_SIZE, -1, TIME_OUT), model2.AI(CHESSBOARD_SIZE, 1, TIME_OUT)]

chessboard = np.zeros((CHESSBOARD_SIZE, CHESSBOARD_SIZE), dtype=np.int8)
chessboard [3, 3] = chessboard [4, 4] = -1
chessboard [3, 4] = chessboard [4, 3] = 1

# chessboard = np.array([
# 	[0,1,1,1,1,1,1,0],
# 	[0,-1,-1,-1,1,-1,-1,0],
# 	[0,0,-1,-1,1,1,0,0],
# 	[0,-1,-1,-1,-1,1,1,0],
# 	[-1,-1,-1,1,-1,-1,1,-1],
# 	[-1,-1,-1,-1,-1,-1,1,1],
# 	[-1,-1,1,-1,-1,0,-1,1],
# 	[-1,-1,-1,-1,-1,0,0,0]
# ], dtype=np.int8)
now_player = 0

while(len(list(model1.get_candidate_list(chessboard,-1)))+len(list(model1.get_candidate_list(chessboard,1)))>0):
	print(chessboard)
	candidate_list = player[now_player].go(chessboard)
	print(candidate_list)
	if(len(candidate_list)>0):
		print(candidate_list[-1])
		chessboard = model1.get_next_board(chessboard,candidate_list[-1], -1+2*now_player)
	print(chessboard)
	print("==========")
	now_player = 1 - now_player

print(model1.judge(chessboard))