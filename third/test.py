from main import *
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -