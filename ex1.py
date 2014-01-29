import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def cost(thetas, X, y):
    m = X.shape[0]
    deltas = (np.dot(X, thetas) - y)
    return np.sum(deltas**2/(2*m)) 

def gradient_descent(thetas, X, y, n, alpha):
    m = X.shape[0]
    costs = []
    for i in range(n):
        deltas = np.dot(X.T, (np.dot(X, thetas) - y))
        thetas = thetas - alpha * deltas / m;
        costs.append(cost(thetas, X, y))
    return thetas, costs

data = pd.read_csv('./ex1/data1.txt', names=['x', 'y'])
m = data.y.shape[0]

plt.plot(data.x, data.y, marker='x', linestyle='None', color='red')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

X = np.array([np.ones(shape=(m, 1)), data.x]).T
thetas = np.array([0, 0])

n = 1500;
alpha = 0.01;

theta, costs = gradient_descent(thetas, X, data.y, n, alpha)

plt.plot(range(n), costs)
plt.ylabel("Cost (Squared Error)")
plt.xlabel("Iteration of Gradient Descent")
plt.show()
