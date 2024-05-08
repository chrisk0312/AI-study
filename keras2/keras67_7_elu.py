import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def mish(x): # Mish: A Self Regularized Non-Monotonic Neural Activation Function
#     return x * np.tanh(np.log(1 + np.exp(x)))

elu = lambda x: x * np.tanh(np.log(1 + np.exp(x)))
y= elu(x)

plt.plot(x, y)
plt.grid()
plt.show()