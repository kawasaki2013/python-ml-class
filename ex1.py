import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def error(t0, t1, x, y):
    return t0 + t1*x - y

def cost(t0, t1, x, y):
    # sum of squared errors
    sse = sum([error(t0, t1, x[i], y[i])**2 for i in range(len(x))])
    return sse/(2*len(x))


data = pd.read_csv('./ex1/data1.txt', names=['x', 'y'])
m = data.y.shape[0]

plt.plot(data.x, data.y, marker='x', linestyle='None', color='red')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
# plt.show()

X = np.array([np.ones(shape=(m, 1)), data.x]).T

t0, t1 = 0, 0
print cost(t0, t1, data.x, data.y)
n = 1500
alpha = 0.01
