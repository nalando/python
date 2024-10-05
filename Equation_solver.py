import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定義方程
def equation(x):
    return [
        np.cos(x[1]) + (x[0] / x[1]) * np.sin(x[1]) - x[2]**(x[0]-1) * (np.cos(x[1]*np.log(x[2])) + (x[0] / x[1]) * np.sin(x[1]*np.log(x[2]))),
        0,  # 佔位符
        0   # 佔位符
    ]

# 設定範圍
x0_vals = np.linspace(1.0, 50.0, 100)  # 在1到50之間均勻取100個點
y_vals = np.linspace(1.0, 10.0, 100)    # 可以根據需要調整範圍
X0, Y = np.meshgrid(x0_vals, y_vals)

# 將 X0, Y 轉換為一維數組以便計算
Z = np.zeros_like(X0)  # 創建一個和 X0 一樣大小的空數組

# 遍歷每個 x0 和 y，計算對應的 z
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        x_guess = np.array([X0[i, j], Y[i, j], 1.0])  # 初始化 x[0] 和 x[1]
        solution = fsolve(equation, x_guess)  # 求解
        Z[i, j] = solution[2]  # 將 z 值存儲在 Z 中

# 繪製三維表面圖
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X0, Y, Z, cmap='viridis')

ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')
plt.title('3D Surface Plot of Solutions')
plt.show()
