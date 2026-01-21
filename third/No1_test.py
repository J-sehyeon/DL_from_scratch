from main import *
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)