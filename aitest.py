import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# 定义方程
def equation(x):
    return np.cos(x[1]) + (x[0] / x[1]) * np.sin(x[1]) - x[2]**(x[0]-1) * (np.cos(x[1]*np.log(x[2])) + (x[0] / x[1]) * np.sin(x[1]*np.log(x[2])))

# 初始猜测值
x_guess = np.array([1.0, 1.0, 1.0])
fx = []
print(type(x_guess[0]))

# 使用Newton-Raphson方法求解方程
i = 0
while x_guess[0] <= 50:
    solution = newton(equation, x_guess)
    x_guess[0] += 0.1
    fx.append(solution)
i = 0
x = []
for i in 100:
    x.append(fx[i][0])
    i += 1
i = 0
y = []
for i in 100:
    y.append(fx[i][0])
    i += 1
i = 0
z = []
for i in 100:
    z.append(fx[i][0])
    i += 1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y , z , c=z1, cmap='Reds', marker='^', label='My Points 1')
ax.legend()
plt.show()
