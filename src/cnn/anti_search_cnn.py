import torch
import torch.nn as nn

chessboard_size = 8
channels_1 = 6
kernal_size_1 = 5
channels_2 = 20
kernal_size_2 = 5
len_fc1 = 100

class EvalNet(nn.Module):
	def __init__(self):
		super(EvalNet, self).__init__()
		self.conv1 = nn.Conv2d(1, channels_1, kernel_size=kernal_size_1, padding=(kernal_size_1-1)//2, padding_mode='circular')
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_size=kernal_size_2, padding=(kernal_size_2-1)//2, padding_mode='circular')
		self.relu2 = nn.ReLU()

		self.fc1 = nn.Linear(chessboard_size * chessboard_size * channels_2, len_fc1)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(len_fc1, 2)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.relu1(self.conv1(x))
		x = self.relu2(self.conv2(x))
		x = x.view(-1, chessboard_size * chessboard_size * channels_2)
		x = self.relu3(self.fc1(x))
		x = self.fc2(x)
		return x