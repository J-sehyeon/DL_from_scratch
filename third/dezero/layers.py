import numpy as np
import weakref

import dezero.functions as F
from dezero.core import Parameter


class Layer:
    def __init__(self):
        self._params = set()                # step44    / Layer 인스턴스에 속한 매개변수(name)를 보관
    
    def __setattr__(self, name, value):     # step44    / obj.x = y를 실행 시 호툴되는 메서드
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)    # step44    / object 상속, __dict__에 저장
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()     # step45    / 제너레이터로 또다른 제너레이터를 만들 때 yield from 사용
            else:
                yield self.__dict__[name]
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:            # in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()
        
        if nobias:
            self.b = None
        
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):   
        if self.W.data is None:             # step44    / 데이터를 흘려보내는 시점에 가중치 초기화
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y