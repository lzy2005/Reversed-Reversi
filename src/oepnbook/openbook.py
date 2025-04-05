from hpc import mcts_hpc_mix as model1 # suc
import numpy as np

CHESSBOARD_SIZE = 8
TIME_OUT = 100

chessboard = np.zeros((CHESSBOARD_SIZE, CHESSBOARD_SIZE), dtype=np.int8)
chessboard[3][3] = chessboard[4][4] = 1
chessboard[3][4] = chessboard[4][3] =-1
now_player = 0

while(len(model1.get_candidate_list(chessboard,-1))+len(model1.get_candidate_list(chessboard,1))>0):
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