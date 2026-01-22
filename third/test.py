from main import *
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -