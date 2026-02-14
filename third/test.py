from dezero import Variable, Parameter, Model
from dezero import optimizers
from dezero.utils import *
from dezero.models import MLP
from tools import plot_2dfunc
import dezero.functions as F
import dezero.layers as L

import numpy as np
import math
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t = np.array([1, 2, 0])
acc = F.accuracy(y, t)
print(acc)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -