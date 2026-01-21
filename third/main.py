import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):          # f = Function() 형태로 함수의 인스턴스를 변수 f에 대입 가능    / input 은 Variable 인스턴스라 가정
        x = input.data                  # 데이터를 꺼낸다
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()     # 구체적 로직은 하위 클래스에서 구현

# Function 하위 클래스
class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

#  Autograd
def numerical_diff(f, x, eps=1e-4):     # 중앙차분 : centered difference
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)