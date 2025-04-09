from hpc import mcts_hpc as model1 # pio
from hpc import mcts_hpc_mix as model2
import numpy as np

CHESSBOARD_SIZE = 8
TIME_OUT = 3

count = 0
model1_win = 0
model2_win = 0
while True:
	chessboard = np.zeros((CHESSBOARD_SIZE, CHESSBOARD_SIZE), dtype=np.int8)
	chessboard [3, 3] = chessboard [4, 4] = -1
	chessboard [3, 4] = chessboard [4, 3] = 1

	if count %2 ==0:
		player = [model1.AI(CHESSBOARD_SIZE, -1, TIME_OUT), model2.AI(CHESSBOARD_SIZE, 1, TIME_OUT)]
	else:
		player = [model2.AI(CHESSBOARD_SIZE, -1, TIME_OUT), model1.AI(CHESSBOARD_SIZE, 1, TIME_OUT)]
	now_player = 0

	while(len(model1.get_candidate_list(chessboard,-1))+len(model1.get_candidate_list(chessboard,1))>0):
		# print(chessboard)
		candidate_list = player[now_player].go(chessboard)
		# print(candidate_list)
		if(len(candidate_list)>0):
			# print(candidate_list[-1])
			chessboard = model1.get_next_board(chessboard,candidate_list[-1], -1+2*now_player)
		# print(chessboard)
		now_player = 1 - now_player

	result = model1.judge(chessboard)
	if count % 2 == 0:
		model1_result = result
	else:
		model1_result = 1-result
	model1_win += model1_result
	model2_win += 1-model1_result

	print(f"round {count + 1} | model1_result {model1_result}")
	print(f"model1_win {model1_win / (count + 1)} | model2_win {model2_win / (count + 1)}")
	count += 1
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