import numpy as np
import weakref
import contextlib
import math         # step27    / my_sin() 함수에 사용


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Variable:
    __array_priority__ = 200            # step21    / ndarray와 Variable 인스턴스의 연산에서 Variable 인스턴스의 연산자 메서드가 우선적으로 호출
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):        # step09    / data의 type을 np.ndarray로 한정
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
            
        self.data = data    
        self.name = name                # step19    / 수많은 변수들을 구분하기 위한 이름
        self.grad = None                # 인스턴스 변수인 data와 grad는 모두 넘파이의 다차원 배열인 ndarray이라고 가정한다.
        self.creator = None             # step07    / 변수의 창조자(함수 혹은 사용자)를 지정, 저장
        self.generation = 0             # step16    / layer와 같은 개념
    
    def set_creator(self, func):        # step07    / creator 인스턴스 변수를 설정하는 메서드
        self.creator = func             # 변수와 함수를 연결한다.
        self.generation = func.generation + 1       # step16    / 함수에 의해 생성된 변수의 세대는 함수의 세대 + 1
    
    def cleargrad(self):                # step14    / 변수 x를 서로 다른 두 계산에 사용할 경우 x.grad를 공유하므로 이를 방지하기 위한 메서드
        self.grad = None
    
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # step09    / y.grad = np.array(1.0) 자동화, data와 grad의 데이터 타입(비트수) 통일

        funcs = []
        seen_set = set()                # step16    / 2개 이상을 리턴하는 함수에 대해 그 함수를 중복해서 funcs리스트에 넣지 않기 위함

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()             # step08    / 리스트에서 마지막 원소를 리턴 후 제거
            gys = [output().grad for output in f.outputs]       # step17    / 함수의 output은 전부 약한 참조이므로 값을 불러오기 위해 "()"를 붙여야한다.
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx    # step14    / +=는 in-place연산이다. 이는 새로운 메모리 위치를 생성하지 않아 주소가 복사되는 위험이 있다.

                if x.creator is not None:
                    add_func(x.creator)     # step16    / 수정 전: func.append(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None         
    
    @property                           # step19    / ndarray 인스턴스 변수
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'
    


class Function:                             # step12    / 가변 길이 변수에 대한 처리 적용
    def __call__(self, *inputs):            # f = Function() 형태로 함수의 인스턴스를 변수 f에 대입 가능    / input 은 Variable 인스턴스라 가정
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]       # step11    / 가변 길이에 대한 처리
        ys = self.forward(*xs)              # step12    / 리스트 언팩 : *xs == x0, x1
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(np.array(y)) for y in ys]

        if Config.enable_backprop:          # step18    / 순전파만 할 경우 역전파를 위한 데이터 저장 x  
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]      # step17    / 약한 참조 도입으로 순환 참조에 의한 메모리 누적을 제거

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

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
         (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

class Sin(Function):                # step27
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data  
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

def newton_f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def newton_gx2(x):
    return 12 * x ** 2 - 4

# 기본 연산자
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y                        # step11    / 튜플 형태로 반환 -> stpe12 : 리스트에서 원소를 뽑아내고 이후에 리스트로 만드는 과정을 상위 클래스에서 지원
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)                   # step21
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0 ,x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

    
# operator overload
Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

#  utils
def numerical_diff(f, x, eps=1e-4):     # 중앙차분 : centered difference
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class Config:                           # step18    / 설정 데이터는 한 군데에만 존재하는게 좋다. 인스턴스 생성 x
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

