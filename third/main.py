import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):        # step09    / data의 type을 np.ndarray로 한정
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
            
        self.data = data    
        self.grad = None                # 인스턴스 변수인 data와 grad는 모두 넘파이의 다차원 배열인 ndarray이라고 가정한다.
        self.creator = None             # step07    / 변수의 창조자(함수 혹은 사용자)를 지정, 저장
    
    def set_creator(self, func):        # step07    / creator 인스턴스 변수를 설정하는 메서드
        self.creator = func             # 변수와 함수를 연결한다.
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # step09    / y.grad = np.array(1.0) 자동화, data와 grad의 데이터 타입(비트수) 통일

        funcs = [self.creator]          # step08    / 반복문 형식의 역전파
        while funcs:
            f = funcs.pop()             # step08    / 리스트에서 마지막 원소를 리턴 후 제거
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):          # f = Function() 형태로 함수의 인스턴스를 변수 f에 대입 가능    / input 은 Variable 인스턴스라 가정
        x = input.data                  # 데이터를 꺼낸다
        y = self.forward(x)
        output = Variable(np.array(y))  # step09    / np.ndarray 연산 이후 np.float으로 변경되어서 Variable의 데이터 타입에서 오류 발생 가능하므로 타입 지정
        output.set_creator(self)        # step07    / 출력 변수에 창조자 설정
        self.input = input              # step06    / 입력 변수 기억(보관)  / 함수의 앞, 뒤 관계 기억
        self.output = output            # step07    / f1(x) = output -> f2(output) = y  여기서 "->"이 관계가 output에 들어있다.
        return output

    def forward(self, x):
        raise NotImplementedError()     # 구체적 로직은 하위 클래스에서 구현
    
    def backward(self, gy):
        raise NotImplementedError

# Function 하위 클래스
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy                 # steo06    / 앞선 노드에서 온 미분값에 자기 자신의 미분값을 곱해 뒤로 보낸다.
        return gx

def square(x):                          # step09    / 함수 사용 편의성 개선
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

#  utils
def numerical_diff(f, x, eps=1e-4):     # 중앙차분 : centered difference
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

