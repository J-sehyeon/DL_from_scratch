from dezero.core import Variable
from dezero.utils import *
from tools import plot_2dfunc
import dezero.functions as F

import numpy as np
import matplotlib.pyplot as plt


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

x = Variable(np.array(2.0))
x.name = 'x'
y = x ** 2
y.nam = 'y'
y.backward(create_graph=True)
gx = x.grad
gx.name = 'gx'
x.cleargrad()

z = gx ** 3 + y
z.name = 'z'

z.backward()
print(x.grad)

print(x.generation, x.grad.generation, gx.generation)

plot_dot_graph(z, verbose=False, to_file="third/chapter/compute_graph/double_back.png")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -