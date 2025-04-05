import numpy as np

# 定义元素范围
a = 5  # 相当于 np.arange(5)，即 [0, 1, 2, 3, 4]
# 定义每个元素被选取的概率
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]

# 按概率选取单个元素
selected_element = np.random.choice(a, p=probabilities)
print("选取的单个元素:", selected_element)