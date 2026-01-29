from dezero.core_simple import Variable
from dezero.utils import *
from main import *
from tools import plot_2dfunc

import numpy as np
import matplotlib.pyplot as plt


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

x = Variable(np.array(2.0))
iters_newton = 10
history_newton = []

for i in range(iters_newton):
    y = newton_f(x)
    x.cleargrad()
    y.backward()

    history_newton.append([x.data.copy(), y.data])

    x.data -= x.grad / newton_gx2(x.data)

x = Variable(np.array(2.0))
iters_gd = 10000
lr = 0.0001
history_gd = []

for i in range(iters_gd):
    y = newton_f(x)
    x.cleargrad()
    y.backward()

    if i % 50 == 0:
        history_gd.append([x.data.copy(), y.data])

    x.data -= lr * x.grad

history_newton = np.array(history_newton)
history_gd = np.array(history_gd)

X = np.arange(-2.2, 2.2, 0.1)
Y = newton_f(X)    

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(X, Y)
ax1.plot(history_newton[:, 0], history_newton[:, 1], marker='o', color="orange", linewidth=2, zorder=20)
ax1.set_title('newton method')

ax2.plot(X, Y)
ax2.plot(history_gd[:, 0], history_gd[:, 1], marker='o', color='orange', linewidth=2, zorder=20)
ax2.set_title('gradient descent method')

plt.legend()
plt.show()




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -