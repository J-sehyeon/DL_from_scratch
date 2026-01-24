from main import *
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
x = Variable(np.array(3))
y = add(add(x, x), x)
y.backward()    
print(x.grad)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -