import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mat = np.identity(5)
data = pd.read_csv('./ex1/data1.txt', names=['x', 'y'])
m = data.y.shape[0]

plt.plot(data.x, data.y, marker='x', linestyle='None', color='red')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()


