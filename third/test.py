from dezero.core_simple import Variable
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

a = Variable(np.array(3.0))
b = 3.0 * a + 1.0
print(b)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -