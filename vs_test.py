"""New Module"""
import numpy as np
import matplotlib.pyplot as plt

xs  = np.linspace(0,20,200)
ys = np.sin(xs)

plt.scatter(xs,ys)
plt.show()
