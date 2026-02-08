from dezero.core import Variable
from dezero.utils import *
from tools import plot_2dfunc
import dezero.functions as F

import numpy as np
import matplotlib.pyplot as plt


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 * x1
print(y)

y.backward()
print(x1.grad)





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -