import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest




# 在數據中人工加入異常值
X[10] = 3
y[10] = 250

# 將數據合併成一個二維數組
data_2d = np.column_stack((X, y))

plt.subplot(211)
# 繪製數據的散點圖
plt.scatter(X, y)
plt.title('Scatter plot of Linear Regression Data')
plt.xlabel('X')
plt.ylabel('Y')


# 使用孤立森林檢測異常值
isolation_forest = IsolationForest(contamination=0.1)
isolation_forest.fit(data_2d)
outliers = isolation_forest.predict(data_2d)

# 獲取異常值的索引
outlier_indices = np.where(outliers == -1)[0]

plt.subplot(212)
if outlier_indices.size > 0:
    print("檢測到異常值，索引位置為：", outlier_indices)
    # 將孤立森林的檢測結果視覺化
    xx, yy = np.meshgrid(np.linspace(X.min(), X.max(), 100), np.linspace(y.min(), y.max(), 100))
    Z = isolation_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.scatter(X, y)
    plt.title('Isolation Forest Decision Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
else:
    print("未檢測到異常值")