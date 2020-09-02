"""
    Tensor implementation of Automatic Differentiation  
    by Joel Grus.

    youtube : https://www.youtube.com/user/joelgrus
    
    github : https://github.com/joelgrus/autograd/tree/part06
    
    Implementations are mostly equal. Some of part modified. 
    Thanks to its videos. I could understand how to implement
    automatic differentiation in python. 

    Joel's tensor ops : 
     - add
     - tensor_sum
     - mul
     - neg
     - matmul
     - sub
     - slice
    
    Added tensor ops by myself :
     - dot
     - power
     - log
     - log_b
     - exp
     - sin 
     - cos
     - tan
     - cot
     - ones
     - zeros
     - div
     - where
     - transpose
     - reshape
     - flatten
     - mean
     - sinh
     - cosh
     - arcsin
     - arccos
     - arctan
     - abs
     - dropout
     - append

    Author : @MGokcayK github.com/MGokcayK
    Create : 24 / 03 / 2020
    Update : 02 / 09 / 2020
                Add append ops.
"""


import numpy as np 
import gNet.conv_utils as conv_util
import gNet.tensor_ops as t_ops
from typing import List, Tuple, NamedTuple, Callable, Optional, Union

class Dependency(NamedTuple):
    '''
        Make dependency class to have gradient function of tensor.
    '''
    tensor: 'Tensor'
    grad_fn : Callable[[np.ndarray], np.ndarray]
    ops_name : 'ops'

# Arrayable types declared.
arrayableTypes = Union[float, list, np.ndarray]

# check whether arrayable is array or not.
# if arrayable is np.ndarray it pass directly
# if not make it array as `np.array`.
def make_array (arrayable: arrayableTypes) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable, dtype=np.float32)

# Tensorable types declared
Tensorable = Union['Tensor', float, int, np.ndarray]

# check whethere tensorable is tensor or not.
# if tensorable is tensor it pass directly
# if not make it tensor.
def make_tensor(tensorable: Tensorable, have_grad=False) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable, have_grad)



class Tensor:
    '''
        Implementation of custom Tensor class. 
        It is based on numpy.ndarray.
    '''

    # Magic Methods
    def __init__(self, 
                value : arrayableTypes,
                have_grad : bool = False,
                depends_on : List[Dependency] = None) -> None:
        self._value = make_array(value)
        self.have_grad = have_grad
        self.depends_on = depends_on or []
        self.grad: Optional['Tensor'] = None

        if self.have_grad:
            self.zero_grad()
            #self.one_grad()

    # handle of numpy's reversed ops like radd, rmat etc. of broadcasting.
    __array_ufunc__ = None

    def __repr__(self):
        return f"Tensor({self._value},shape=({self.shape}), have_grad={self.have_grad})"

    def __add__(self, other) -> 'Tensor':
        '''
            Addition when user used `+` operator.
            Ex : t + other.
        '''
        return add(self, make_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        '''
            Reverse addition when user used `+` operator.
            Ex : other + t.
        '''
        return add(make_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        '''
            Addition when user used `+=` operator.
            Ex : t += other  ==  t = t + other.
        '''
        return add(self, make_tensor(other))

    def __sub__(self, other) -> 'Tensor':
        '''
            Substraction when user used `-` operator.
            Ex : t - other.
        '''
        return sub(self, make_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        '''
            Reverse substraction when user used `-` operator.
            Ex : other - t.
        '''
        return sub(make_tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        '''
            Substraction when user used `-=` operator.
            Ex : t -= other  ==  t = t - other.
        '''
        return sub(self, make_tensor(other))

    def __mul__(self, other) -> 'Tensor':
        '''
            Element wise matrix multiplication when user used `*` operator.
            Ex : t * other.
        '''
        return mul(self, make_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        '''
            Reverse element wise matrix multiplication when user used `*` operator.
            Ex : other * t. 
        '''
        return mul(make_tensor(other), self)

    def __imul__(self, other) -> 'Tensor':
        '''
            Element wise matrix multiplication when user used `*=` operator.
            Ex : t *= other  ==  t = t * other.
        '''
        return mul(self, make_tensor(other))

    def __matmul__(self, other) -> 'Tensor':
        '''
            Matrix multiplication when user used `@` operator.
            Ex : t @ other.
        '''
        return matmul(self, make_tensor(other))
    
    def __rmatmul__(self, other) -> 'Tensor':
        '''
            Reverse matrix multiplication when user used `@` operator.
            Ex : other @ t.
        '''
        return matmul(make_tensor(other), self)
    
    def __imatmul__(self, other) -> 'Tensor':
        '''
            Matrix multiplication when user used `@=` operator.
            Ex : t @ other == t = t @ other.
        '''
        return matmul(self, make_tensor(other))
    
    def __truediv__(self, other) -> 'Tensor':
        '''
            Element wise matrix division when user used `/` operator.
            Ex : t / other.
        '''
        return div(self, make_tensor(other))

    def __rtruediv__(self, other) -> 'Tensor':
        '''
            Reverse element wise matrix division when user used `/` operator.
            Ex : other / t.
        '''
        return div(make_tensor(other), self)

    def __itruediv__(self, other) -> 'Tensor':
        '''
            Element wise matrix division when user used `/=` operator.
            Ex : t /= other  ==  t = t / other.
        '''
        return div(self, make_tensor(other))
  
    def __floordiv__(self, other) -> 'Tensor':
        '''
            Element wise matrix division when user used `//` operator.
            Ex : t // other.
        '''
        return div(self, make_tensor(other))

    def __rfloordiv__(self, other) -> 'Tensor':
        '''
            Reverse element wise matrix division when user used `//` operator.
            Ex : other // t.
        '''
        return div(make_tensor(other), self)

    def __idiv__(self, other) -> 'Tensor':
        '''
            Element wise matrix division when user used `/=` operator.
            Ex : t //= other  ==  t = t // other.
        '''
        return div(self, make_tensor(other))

    def __neg__(self) -> 'Tensor':
        '''
            Negative of tensor. 
            Ex : -t 
        '''
        return neg(self)

    def __pow__(self, other) -> 'Tensor':
        '''
            Power of tensor when `**` operator used.
        '''
        return power(self, other)

    def __ipow__(self, other) -> 'Tensor':
        '''
            Power of tensor when `**=` operator used.
        '''
        return power(self, other)

    def __getitem__(self, idxs) -> 'Tensor':
        return tensor_slice(self, idxs)

    # Other Methods
    def zero_grad(self) -> None:
        '''
            If tensor have gradient, make it zero as initial value.
        '''
        self.grad = make_tensor(np.zeros_like(self._value, dtype=np.float32))


    def one_grad(self) -> None:
        self.grad = Tensor(np.ones(self._value.shape, dtype=np.float64))


    def backward(self, grad: 'Tensor' = None, ops_name=None) -> None:
        '''
            Calculate backward propagation of tensor.
        '''
        assert self.have_grad, "Called backward propagation of non-gradientable tensor. \n\
                                Make sure that tensor has `have_grad=True` ."
        
        if grad is None:
            grad = ones(self.shape)

        grad = make_tensor(grad)

        self.grad._value = grad._value + self.grad._value

        for dependency in self.depends_on:
            if isinstance(dependency, Dependency):
                back_grad = dependency.grad_fn(grad._value)
                ops_name = dependency.ops_name
                dependency.tensor.backward(Tensor(back_grad), ops_name)
            

    def sum(self, axis=None, keepdim=False) -> 'Tensor':
        '''
            Summation of 0-tensor of tensor.
            Ex: t = Tensor([1,3,4], [2,4,1])
                t.sum() == 15
        '''
        return tensor_sum(self, axis=axis, keepdim=keepdim)

    # Properties 
    @property
    def value(self) -> np.ndarray:
        '''
            Return value of Tensor.
        '''
        return self._value

    @value.setter
    def value(self, new_value: np.ndarray)-> None:
        '''
            Set value of Tensor manually.
        '''
        self._value = make_array(new_value)

    @property
    def shape(self):
        return self._value.shape


# Tensor operations.

def tensor_sum(t: Tensor, axis=0, keepdim=False) -> Tensor:
    """
        Summation of element of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t    : Tensor

        axis : Summation axis.
            >>> type    : integer
            >>> Default : 0

        keepdim : Keeping dimension of tensor.
            >>> type    : bool
            >>> Default : False

        Sum tensor w.r.t axis. 
            If axis=0 mean 0-tensor.
            If axis=1 mean sum along axis = 1
            If axis=2 mean sum along axis = 2
       
    """
    return t_ops.tensor_sum(make_tensor(t), axis, keepdim)



def add(t1: Tensor, t2: Tensor) -> Tensor:
    """
        Addition of two `Tensor`. Also it is calculate its gradient of
        operation if one of tensor's have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = a + b 

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = a**2 + b \n
        x = a**3

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a.

        If a.have_grad = True => a.grad can be calculated by calling x.backward()
        and result equal to 3a**2.

        If a.have_grad = True => a.grad can be calculated by calling y.backward() and x.backward()
        and result equal to 2a + 3a**2.      

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.add(t1, t2)
    
    
    
def mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
        Element wise multiplication of two `Tensor`. Also it is calculate its gradient of
        operation if one of tensor's have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = a * b 

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = a**2 * b \n

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2ab * a'.

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.mul(make_tensor(t1), make_tensor(t2))


    
def div(t1: Tensor, t2: Tensor) -> Tensor:
    """
        Element wise division of two `Tensor`. Also it is calculate its gradient of
        operation if one of tensor's have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = a / b 

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = a**2 / b \n

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a/b * a'.    

        If b.have_grad = True => b.grad can be calculated by calling y.backward()
        and result equal to a**2/b**2 * b'.    

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """    
    return t_ops.div(make_tensor(t1), make_tensor(t2))



def neg(t: Tensor)-> Tensor:
    """
        Negative of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t : Tensor

        For example:
        -----------

        y = -a + b = neg(a) + b

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = -(a**2) + b = neg(a**2) + b

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to -2a.  

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.neg(make_tensor(t))



def sub(t1: Tensor, t2: Tensor) -> Tensor:
    """
        Substraction of two `Tensor`. Also it is calculate its gradient of
        operation if one of tensor's have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = b - a

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = b - a**2 \n

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to -2a.   

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return add(make_tensor(t1), neg(make_tensor(t2)))



def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
        Matrix multiplication of two `Tensor`. Also it is calculate its gradient
        of operatation if tensor have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = a @ b 

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        If t1 shape (n1, m1) and t2 is (m1, m2), then t3 which is t1 @ t2 is (n1, m2)
        Thus, t3.grad is also (n1, m2)

        So, 
            t1.grad = t3.grad @ t2.T  ==> (n1,m2) (m2, m1) => (n1,m1)
            t2.grad = t1.T @ t3.grad  ==> (m1,n1) (n1, m2) => (m1,m2)
    """
    return t_ops.matmul(make_tensor(t1), make_tensor(t2))



def tensor_slice(t: Tensor, idxs) -> Tensor:
    """
        Slicing of `Tensor`. Also it is calculate its gradient
        of operatation if tensor have_grad = True.

        Arguments:
        ----------

        t    : Tensor

        idxs : Index

        For example:
        -----------

        a = [a_0, a_1, ... , a_n]

        y = a[1:3] = [a_1, a_2]

        dy/da = [0, a_1', a_2', 0, ... , 0]

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        a.grad calculated only indexed elements.

    """
    return t_ops.tensor_slice(make_tensor(t), idxs)



def power(t: Tensor, p=1) -> Tensor:
    """
        Power of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t   : Tensor

        p   : power
            >>> type    : integer
            >>> Default : 1

        For example:
        -----------

        y = a ** 3 + b ** 2

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = a**3 + b ** 2 \n

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 3a**2.

        If b.have_grad = True => b.grad can be calculated by calling y.backward()
        and result equal to 2b.

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.power(make_tensor(t), p)



def log(t: Tensor) -> Tensor:
    """
        Natural logarithm (base e) of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = log(a) + log(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = log(a) + log(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to a'/a.

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.log(make_tensor(t))



def log_b(t: Tensor, b = 10) -> Tensor:
    """
        Logarithm of base b of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t   : Tensor

        b   : Base
            >>> type    : integer
            >>> Default : 10

        For example:
        -----------

        y = (log_3)(a) + (log_7)(b) = log(a, 3) + log(b, 7)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = (log_3)(a) + (log_7)(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to a'/(log(3) * a).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.log_b(make_tensor(t), b)



def exp(t: Tensor) -> Tensor:
    """
        Exponential of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = e^a + e^b = exp(a) + exp(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = e^(a**2+2a)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to (2a + 2) * e^(a**2+2a).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.exp(make_tensor(t))



def sin(t: Tensor) -> Tensor:
    """
        Sinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Sinus in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = sin(a) + sin (b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = sin(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a * cos(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.sin(make_tensor(t))



def arcsin(t: Tensor) -> Tensor:
    """
        Arcsinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arcsinus in radian. Tensor elements should be in [-pi/2, pi/2].

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = arcsin(a) + arcsin(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = arcsin(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a / (1 - (a**2)**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.arcsin(make_tensor(t))



def cos(t: Tensor) -> Tensor:
    """
        Cosinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Cosinus in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = cos(a) + cos(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = cos(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to -2a * sin(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.cos(make_tensor(t))



def arccos(t: Tensor) -> Tensor:
    """
        Arccosinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arccosinus in radian. Tensor elements should be in [-1, 1].

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = arcsin(a) + arcsin(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = arcsin(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to  -2a / (1 - (a**2)**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.arccos(make_tensor(t))



def tan(t: Tensor) -> Tensor:
    """
        Tanjent of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Tangent in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = tan(a) + tan(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = tan(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a * 1/cos^2(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.tan(make_tensor(t))



def arctan(t: Tensor) -> Tensor:
    """
        Arctangent of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arctangent in radian. Tensor elements should be in [-pi/2, pi/2].

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = arctan(a) + arctan(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = arctan(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a / (1 + (a**2)**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.arctan(make_tensor(t))



def cot(t: Tensor) -> Tensor:
    """
        Cotanjent of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Cotangent in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = cot(a) + cot(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = cot(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to -2a * 1/sin^2(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.cot(make_tensor(t))


    
def ones(shape=None, have_grad=False) -> Tensor:
    """
        Creating ones of `Tensor`.

        Arguments:
        ----------

        shape           : Shape of ones tensor
            >>> type    : tuple
            >>> Default : None

        have_grad       : Assign whether tensor has gradient or not.
            >>> tpye    : bool
            >>> Default : False

        For example:
        -----------
        
        y = ones((2,2), True)

        y = [[1,1],[1,1]] and y.grad can be calculated.
    """
    return make_tensor(np.ones(shape), have_grad)
    


def zeros(shape=None, have_grad=False) -> Tensor:
    """
        Creating zeros of `Tensor`.

        Arguments:
        ----------

        shape           : Shape of ones tensor
            >>> type    : tuple
            >>> Default : None

        have_grad       : Assign whether tensor has gradient or not.
            >>> tpye    : bool
            >>> Default : False

        For example:
        -----------
        
        y = ones((2,2), True)

        y = [[0,0],[0,0]] and y.grad can be calculated.
    """
    return make_tensor(np.zeros(shape), have_grad)



def mean(t: Tensor, axis=None, keepdim=False) -> Tensor:
    """
        Mean of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Calculation of gradient is division of gradient of graph by tensor size. 

        Arguments:
        ----------

        t    : Tensor

        axis : Calculation of mean axis.
            >>> type    : integer
            >>> Default : None

        keepdim : Keeping dimension.
            >>> type    : bool
            >>> Default : False

        For example:
        -----------

        a = [1,4,8]

        y = mean(a) = 4.3333

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = cot(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to a'/3.

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.mean(make_tensor(t), axis, keepdim)



def where(t: Tensor, condition:None, _true:None, _false:None) -> Tensor:
    """
        Partial function of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Careful:
        --------

        Tensor class has no comparision future directly. Therefore, when assign condition,
        use tensor value instead of tensor itself.

        Ex : y = where(a, a.value > 0, a, 0)
                          
        Arguments:
        ----------

        t         : Tensor

        condition : Condition of tensor element.
            >>> type    : comparision (>, >=, <, <=, ...)
            >>> Default : None

        _true     : Returning of conditionally true value.
            >>> type    : comparision (>, >=, <, <=, ...)
            >>> Default : None

        _false    : Returning of conditionally false value.
            >>> type    : comparision (>, >=, <, <=, ...)
            >>> Default : None

        For example:
        -----------

        a = Tensor([[5.,1.],[2.,4.]], have_grad=True)

        y = where(a, a.value>3, a**2, 2*a)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        a = Tensor([[5.,1.],[2.,4.]], have_grad=True)

        y = where(a, a.value>3, a**2, 2*a)

        y = [[25.  2.]
             [ 4. 16.]]

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to [[10.  2.]
                             [ 2.  8.]].

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.where(make_tensor(t), condition, _true, _false)


   
def reshape(t: Tensor, shape=None) -> Tensor:
    """
        Reshaping of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Calculation of gradient is reshaping gradient w.r.t first shape.

        Arguments:
        ----------

        t         : Tensor

        shape     : New shape of tensor
            >>> type    : tuple
            >>> Default : None

        For example:
        -----------

        a = Tensor(np.random.randn(3,5,6,2), have_grad=True)

        y = reshape(a, (36,5))

        print(y.shape) # prints (36,5)
        print(y.grad.shape) # prints (3,5,6,2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result's shape equal to (3,5,6,2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.reshape(make_tensor(t), shape)



def flatten(t: Tensor, batching=False) -> Tensor:
    """
        Flattening of tensor. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        If tensor has batch, batching argument should be True. Than, it can 
        calculate flatten operation w.r.t batch size. 

        Arguments :
        -----------

        t           : Tensor object which flatten operation will applied on.

        batching    : Tensor batch condition. If tensor has batch size it should be True.
            >>> type    : bool
            >>> Default : False

        For example:
        -----------

        a = Tensor(np.random.randn(3,5,6,2), have_grad=True)

        y = flatten(a, True)

        print(y.shape) # prints (3,60)
        print(y.grad.shape) # prints (3,5,6,2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result's shape equal to (3,5,6,2).

        For example:
        ------------

        a = Tensor(np.random.randn(3,5,6,2), have_grad=True)

        y = flatten(a, False)

        print(y.shape) # prints (180,)
        print(y.grad.shape) # prints (3,5,6,2)

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.flatten(make_tensor(t), batching)



def transpose(t: Tensor, axes=None) -> Tensor:
    """
        Transposing of tensor. Also it is calculate its gradient of
        operation if tensor's have_grad = True. 

        Calculation of gradient is transposing gradient w.r.t first shape.

        Arguments :
        -----------

        t       : Tensor object which flatten operation will applied on.

        axes    : Transposing axes. Order is important.
            >>> type    : tuple
            >>> Default : None

        For example:
        -----------

        a = Tensor(np.random.randn(3,5,6,2), have_grad=True)

        y = transpose(a, (3,2,0,1))

        print(y.shape) # prints (2,6,3,5)
        print(y.grad.shape) # prints (3,5,6,2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result's shape equal to (3,5,6,2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.transpose(make_tensor(t), axes)



def dot(t1: Tensor, t2: Tensor, axes=None) -> Tensor:
    """
        Dot product of two `Tensor`. Also it is calculate its gradient
        of operatation if tensor have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        For example:
        -----------

        y = dot(a, b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        If t1 shape (n1, m1) and t2 is (m1, m2), then t3 which is dot(t1, t2) is (n1, m2)
        Thus, t3.grad is also (n1, m2)

        So, 
            t1.grad = dot(t3.grad, t2.T)  ==> (n1,m2) (m2, m1) => (n1,m1)
            t2.grad = dot(t1.T, t3.grad)  ==> (m1,n1) (n1, m2) => (m1,m2)

        Note:
        -----
        It is very similar to matmul ops. Dot is more quicker.
    """
    return t_ops.dot(make_tensor(t1), make_tensor(t2), axes)



def sinh(t: Tensor) -> Tensor:
    """
        Hyperbolic Sinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Hyperbolic Sinus in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = sinh(a) + sinh(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = sin(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a * cosh(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.sinh(make_tensor(t))



def cosh(t: Tensor) -> Tensor:
    """
        Hyperbolic Cosinus of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Hyperbolic Cosinus in radian.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = cosh(a) + cosh(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = cosh(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a * sin(a**2).

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.cosh(make_tensor(t))



def abs(t: Tensor) -> Tensor:
    """
        Absolute values of elements of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Arguments:
        ----------

        t   : Tensor

        For example:
        -----------

        y = abs(a) + abs(b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        y = abs(a**2)

        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and result equal to 2a if a >=0. If a < 0, result equal to -2a.

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.abs(make_tensor(t))



def dropout(t: Tensor, p = 0.) -> Tensor:
    """
        Dropout operation of `Tensor`. Also it is calculate its gradient of
        operation if tensor's have_grad = True.

        Calculation of gradients is same nodes of feed forward. 
        
        https://arxiv.org/abs/1207.0580
        
        https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network

        Arguments:
        ----------

        t   : Tensor

        p   : Rate of dropout
            >>> type    : float
            >>> Default : 0.

        For example:
        -----------

        y = dropout(a, 0.5)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 

        For example:
        ------------

        a = Tensor([[5.,1.],[3., 4.]], have_grad=True)

        y = dropout(a, 0.5)

        print(y) # one of result is Tensor([[10.  0.]
                                            [ 0.  8.]],shape=((2, 2)), have_grad=True) 
 
        If a.have_grad = True => a.grad can be calculated by calling y.backward()
        and one of result equal to Tensor([[2. 0.]
                                          [0. 2.]],shape=((2, 2)), have_grad=False)

        Note: 
        -----
        Partial derivative is depend on calling method of `backward` like y.backward(). 
    """
    return t_ops.dropout(make_tensor(t), p)


def append(t1: Tensor, t2: Tensor, axis=None) -> Tensor:
    """
        Appending two `Tensor`. Also it is calculate its gradient
        of operatation if tensor have_grad = True.

        Arguments:
        ----------

        t1 : Tensor

        t2 : Tensor

        axis : Append axis

        For example:
        -----------

        y = append(a, b)

        If a.have_grad = True => a.grad can be calculated by calling y.backward().
        It is same for b. 
    """
    return t_ops.append(make_tensor(t1), make_tensor(t2), axis)

