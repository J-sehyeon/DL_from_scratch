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
    
    def cleargrad(self):                # step14    / 변수 x를 서로 다른 두 계산에 사용할 경우 x.grad를 공유하므로 이를 방지하기 위한 메서드
        self.grad = None
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # step09    / y.grad = np.array(1.0) 자동화, data와 grad의 데이터 타입(비트수) 통일

        funcs = [self.creator]          # step08    / 반복문 형식의 역전파
        while funcs:
            f = funcs.pop()             # step08    / 리스트에서 마지막 원소를 리턴 후 제거
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx    # step14    / +=는 in-place연산이다. 이는 새로운 메모리 위치를 생성하지 않아 주소가 복사되는 위험이 있다.

                if x.creator is not None:
                    funcs.append(x.creator)

class Function:                             # step12    / 가변 길이 변수에 대한 처리 적용
    def __call__(self, *inputs):            # f = Function() 형태로 함수의 인스턴스를 변수 f에 대입 가능    / input 은 Variable 인스턴스라 가정
        xs = [x.data for x in inputs]       # step11    / 가변 길이에 대한 처리
        ys = self.forward(*xs)              # step12    / 리스트 언팩 : *xs == x0, x1
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

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
        x = self.inputs[0].data
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

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y                      # step11    / 튜플 형태로 반환 -> stpe12 : 리스트에서 원소를 뽑아내고 이후에 리스트로 만드는 과정을 상위 클래스에서 지원
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)
    

#  utils
def numerical_diff(f, x, eps=1e-4):     # 중앙차분 : centered difference
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

