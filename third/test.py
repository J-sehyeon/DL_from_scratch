from main import *
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
x = Variable(np.array([1, 2, 3]))
print(x)

x = Variable(None)
print(x)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -