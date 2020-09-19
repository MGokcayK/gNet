"""
    Tensor operations implementations.

    Author : @MGokcayK github.com/MGokcayK
    Create : 24 / 03 / 2020
    Update : 19 / 09 / 2020
                Readding mistaken removed sinh ops.
"""

import numpy as np
from gNet import tensor as T


def add(t1: 'Tensor', t2:'Tensor') -> 'Tensor':
    '''
        Addition of two `Tensor`. Also it is calculate its gradient of operation 
        if one of tensor have_grad = True.
    '''
    value = np.add(t1._value, t2._value, dtype=np.float32)
    have_grad = t1.have_grad or t2.have_grad
    ops_name = '_add'
    depends_on: List[Dependency] = []

    if t1.have_grad:
        def grad_fn_add1(grad: np.ndarray) -> np.ndarray:

            # to handle broadcast, add dimension
            ndims_added = grad.ndim - t1._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)
                
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)
                    
            return grad
        
        ops_name = '_add1'
        depends_on.append(T.Dependency(t1, grad_fn_add1, ops_name))

    if t2.have_grad:
        def grad_fn_add2(grad: np.ndarray) -> np.ndarray:

            # to handle broadcast, add dimension
            ndims_added = grad.ndim - t2._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad
        ops_name = '_add2'
        depends_on.append(T.Dependency(t2, grad_fn_add2, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def tensor_sum(t: 'Tensor', axis=0, keepdim=False) -> 'Tensor':
    '''
        Sum tensor w.r.t axis. 
            If axis=0 mean 0-tensor.
            If axis=1 mean sum along axis = 1
            If axis=2 mean sum along axis = 2

        Default axis is 0.
    '''
    value = np.sum(t.value, axis=axis, keepdims=keepdim, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_tensor_sum'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_sum(grad: np.ndarray) -> np.ndarray:
            '''
                Gradient should be 0-tensor. So each element has that much
                gradient.
            '''
            # to handle broadcast, add dimension
            ndims_added = t.value.ndim - grad.ndim 
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)

            for i, dim in enumerate(t.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad.astype(np.float32) * np.ones_like(t._value, dtype=np.float32)
            

        depends_on.append(T.Dependency(t, grad_fn_sum, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def mul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    '''
        Element wise multiplication of two `Tensor`. Also it is calculate its 
        gradient of operation if tensor have_grad = True.
    '''
    value = np.multiply(t1._value, t2._value, dtype=np.float32)
    have_grad = t1.have_grad or t2.have_grad
    ops_name = '_mul'
    depends_on: List[Dependency] = []

    if t1.have_grad:
        def grad_fn_mul1(grad: np.ndarray) -> np.ndarray:
            
            grad = np.multiply(grad, t2._value, dtype=np.float32)

            # to handle broadcast, add dimension
            ndims_added = grad.ndim - t1._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad
        ops_name = '_mul1'
        depends_on.append(T.Dependency(t1, grad_fn_mul1, ops_name))

    if t2.have_grad:
        def grad_fn_mul2(grad: np.ndarray) -> np.ndarray:

            grad = np.multiply(grad, t1._value, dtype=np.float32)

            ndims_added = grad.ndim - t2._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad
        ops_name = '_mul2'
        depends_on.append(T.Dependency(t2, grad_fn_mul2, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def div(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    '''
        Element wise division of two `Tensor`. Also it is calculate its 
        gradient of operation if tensor have_grad = True.
    '''
    value = np.divide(t1._value, (t2._value + 1e-10), dtype=np.float32)
    have_grad = t1.have_grad or t2.have_grad
    ops_name = '_div'

    depends_on: List[Dependency] = []

    if t1.have_grad:
        def grad_fn_div1(grad: np.ndarray) -> np.ndarray:
            
            grad = np.divide(grad, (t2._value + 1e-7), dtype=np.float32)

            # to handle broadcast, add dimension
            ndims_added = grad.ndim - t1._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0, dtype=np.float32)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad
            
        depends_on.append(T.Dependency(t1, grad_fn_div1, ops_name))

    if t2.have_grad:
        def grad_fn_div2(grad: np.ndarray) -> np.ndarray:

            grad =np.divide( -(grad * t1._value), ((t2._value ** 2) + 1e-7), dtype=np.float32)

            #grad = grad / ((t2._value ** 2) + 1e-7)

            ndims_added = grad.ndim - t2._value.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0,dtype=np.float32)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True, dtype=np.float32)

            return grad
        
        depends_on.append(T.Dependency(t2, grad_fn_div2, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def neg(t: 'Tensor')-> 'Tensor':
    '''
        Negative of `Tensor`. Also it is calculate its gradient of operation 
        if tensor have_grad = True.
    '''
    value = -t._value
    have_grad = t.have_grad
    ops_name = '_neg'

    if have_grad:
        depends_on = [T.Dependency(t, lambda x: -x, ops_name)]
    else:
        depends_on = []

    return T.Tensor(value, have_grad, depends_on)



def matmul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    '''
        Matrix multiplication of two `Tensor`. Also it is calculate its gradient
        of operatation if tensor have_grad = True.

        If t1 shape (n1, m1) and t2 is (m1, m2), then t3 which is t1 @ t2 is (n1, m2)
        Thus, t3.grad is also (n1, m2)

        So, 
            t1.grad = t3.grad @ t2.T  ==> (n1,m2) (m2, m1) => (n1,m1)
            t2.grad = t1.T @ t3.grad  ==> (m1,n1) (n1, m2) => (m1,m2)
    '''
    value = np.matmul(t1._value, t2._value, dtype=np.float32)
    have_grad = t1.have_grad or t2.have_grad
    ops_name = '_matmul'

    depends_on: List[Dependency] = []

    if t1.have_grad:
        def grad_fn_matmul1(grad: np.ndarray) -> np.ndarray:
            return np.matmul(grad, t2._value.T, dtype=np.float32)
        ops_name = '_matmul1'
        depends_on.append(T.Dependency(t1, grad_fn_matmul1, ops_name))

    if t2.have_grad:
        def grad_fn_matmul2(grad: np.ndarray) -> np.ndarray:
            return np.matmul(t1._value.T, grad, dtype=np.float32)
        ops_name = '_matmul2'        
        depends_on.append(T.Dependency(t2, grad_fn_matmul2, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def tensor_slice(t: 'Tensor', idxs) -> 'Tensor':
    value = t.value[idxs]
    have_grad = t.have_grad
    ops_name = '_slice'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_slice(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.value, dtype=np.float32)
            bigger_grad[idxs] = grad.astype(np.float32)
            return bigger_grad

        depends_on.append(T.Dependency(t, grad_fn_slice, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def power(t: 'Tensor', p) -> 'Tensor':
    '''
        Power calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True..
    '''    
    value = np.power(t._value, p, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_pow'

    depends_on: List[Dependency] = []
    
    if have_grad:
        def grad_fn_pow(grad: np.ndarray) -> np.ndarray:
            if p == 0:
                grad = 0
            elif p < 0:
                grad = np.multiply(np.multiply(p, np.divide(1., (np.power(t._value, np.absolute(p-1))))), grad.astype(np.float32))
            else:
                grad = np.multiply(np.multiply(p, np.power(t._value, (p-1))), grad.astype(np.float32))
            return grad            

        depends_on.append(T.Dependency(t, grad_fn_pow, ops_name))

    else:
        depends_on = []

    return T.Tensor(value, have_grad, depends_on)



def log(t: 'Tensor') -> 'Tensor':
    '''
        Log (also ln) calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.
    '''
    value = np.log(t._value + 1e-10, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_log'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_log(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(np.divide(1., (t._value + 1e-10),dtype=np.float32), grad, dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_log, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def log_b(t: 'Tensor', b: int) -> 'Tensor':
    '''
        Log of base b calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.
    '''
    value = np.divide(np.log(t._value + 1e-10, dtype=np.float32), np.log(b + 1e-10, dtype=np.float32), dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_log'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_log(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(np.divide(1., np.multiply(t._value, np.log(b,dtype=np.float32) + 1e-10, dtype=np.float32),dtype=np.float32), grad, dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_log, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def exp(t: 'Tensor') -> 'Tensor':
    '''
        Exponent calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.
    '''
    value = np.exp(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_exp'

    if have_grad:
        def grad_fn_exp(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(value, grad, dtype=np.float32)
            return grad

        depends_on = [T.Dependency(t, grad_fn_exp, ops_name)]
    else:
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def sin(t: 'Tensor') -> 'Tensor':
    '''
        Sinus calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Sinus in radian.
    '''
    value = np.sin(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_sin'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_sin(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(np.cos(t._value, dtype=np.float32), grad)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_sin, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def arcsin(t: 'Tensor') -> 'Tensor':
    '''
        Arcinus calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Arcsinus in radian. Tensor should be in range [-pi/2, pi/2]
    '''
    assert np.all(t._value >= -np.pi/2) and np.all(t._value <= np.pi/2), \
        'Tensor value is not in rage which is -pi/2 <= value <= pi/2'
    value = np.arcsin(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_arcsin'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_arcsin(grad: np.ndarray) -> np.ndarray:
            grad = np.power(np.divide(grad, (1. - np.power(t._value, 2, dtype=np.float32))), 0.5)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_arcsin, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def cos(t: 'Tensor') -> 'Tensor':
    '''
        Cosinus calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Cosinus in radian.
    '''
    value = np.cos(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_cos'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_cos(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(-np.sin(t._value), grad)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_cos, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def arccos(t: 'Tensor') -> 'Tensor':
    '''
        Arccosinus calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Arccosinus in radian.
    '''
    assert np.all(t._value >= -1.) and np.all(t._value <= 1.), \
        'Tensor value is not in rage which is -1 <= value <= 1'
    value = np.arccos(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_arccos'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_arccos(grad: np.ndarray) -> np.ndarray:
            grad = np.power(np.divide(- grad, (1. - np.power(t._value, 2, dtype=np.float32)), dtype=np.float32), 0.5, dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_arccos, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def tan(t: 'Tensor') -> 'Tensor':
    '''
        Tangent calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Tangent in radian.
    '''
    value = np.tan(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_tan'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_tan(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(np.divide(1., np.power(np.cos(t._value, dtype=np.float32), 2, dtype=np.float32) + 1e-10 ,dtype=np.float32), grad, dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_tan, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def arctan(t: 'Tensor') -> 'Tensor':
    '''
        Arctangent calculation of tensor. Also it is calculate its gradient of operation 
        if tensor have_grad = True.

        Arctangent in radian. Tensor should be in range [-pi/2, pi/2]
    '''
    value = np.arctan(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_arctan'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_arctan(grad: np.ndarray) -> np.ndarray:
            grad = np.divide(grad, (1. + np.power(t._value, 2, dtype=np.float32)), dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_arctan, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def cot(t: 'Tensor') -> 'Tensor':
    '''
        Cotangent calculation of tensor. Also it is calculate its gradient of operation         if tensor have_grad = True.

        Cotangent in radian.
    '''
    value = np.cot(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_cot'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_cot(grad: np.ndarray) -> np.ndarray:
            grad = np.multiply(-np.divide(1., np.power(np.sin(t._value), 2, dtype=np.float32) + 1e-10, dtype=np.float32), grad, dtype=np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_cot, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def mean(t: 'Tensor', axis=None, keepdim=False) -> 'Tensor':
    value = np.mean(t.value, axis=axis, keepdims=keepdim, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_mean'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_mean(grad: np.ndarray) -> np.ndarray:
            if axis == None:
                return np.divide(grad, t.value.size, dtype=np.float32)
            else:
                ones = np.ones(t.value.shape, dtype=np.float32) 
                ax = axis
                ones = np.divide(np.multiply(ones, np.expand_dims(grad,ax), dtype=np.float32), t.value.size)
                return ones

        depends_on.append(T.Dependency(t, grad_fn_mean, ops_name))

    
    return T.Tensor(value, have_grad, depends_on)



def where(t: 'Tensor', condition:None, _true:None, _false:None) -> 'Tensor':
    '''
        Return condition. 
        If condition true, return _true.
        If condition false, return _false.
    '''
    # if _true or _false are not tensor, convert it to tensor.
    _true = T.make_tensor(_true)
    if _true.grad == None:
        _true.grad = T.Tensor(np.zeros_like(t.value, dtype=np.float32))
    _false = T.make_tensor(_false)
    if _false.grad == None:
        _false.grad = T.Tensor(np.zeros_like(t.value, dtype=np.float32))

    value = np.where(condition, _true.value, _false.value)
    have_grad = _true.have_grad or _false.have_grad
    ops_name = '_where'

    depends_on: List[Dependency] = []

    if _true.have_grad:
        def grad_fn_whereT(grad: np.ndarray) -> np.ndarray:
            return grad.astype(np.float32) * np.where(condition, 1, 0)
        
        depends_on.append(T.Dependency(_true, grad_fn_whereT, ops_name))

    if _false.have_grad:
        def grad_fn_whereF(grad: np.ndarray) -> np.ndarray:
            return grad.astype(np.float32) * np.where(condition, 0, 1)
        
        depends_on.append(T.Dependency(_false, grad_fn_whereF, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def reshape(t: 'Tensor', shape=None) -> 'Tensor':
    '''
        Return maximum of `Tensor`'s element.
    '''
    pre_shape = t.shape
    value = np.reshape(t.value, newshape=shape)
    have_grad = t.have_grad
    ops_name = '_reshape'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_reshape(grad: np.ndarray) -> np.ndarray:
            return np.reshape(grad, pre_shape)

        depends_on.append(T.Dependency(t, grad_fn_reshape, ops_name))

    else: 
        depends_on = []
        
    return T.Tensor(value, have_grad, depends_on)



def flatten(t: 'Tensor', batching=False) -> 'Tensor':
    """
        Flattening of tensor. 
    """
    # if flatten operation has batch it means that the operation in
    # training. Therefore, it should have batch_size and batch_size increase
    # dimension of tensor. To handle flattening operation with batch_size,
    # value should be reshaped w.r.t batch_size.
    # on the other hand, when batching is False, it means that the operation
    # called for non-training condition such as calculating formula. Thus,
    # it just directly flatten the tensor without batch_size.
    if batching:
        batch_size = t.shape[0]
        value = t._value.reshape(batch_size, -1)
    else:
        value = t._value.flatten()
    grad_shape = t.shape
    have_grad = t.have_grad
    ops_name = '_flatten'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_flatten(grad: np.ndarray):
            return grad.reshape(grad_shape)

        depends_on.append(T.Dependency(t, grad_fn_flatten, ops_name))

    return T.Tensor(value, have_grad, depends_on)



def transpose(t: 'Tensor', axes=(1,0)) -> 'Tensor':
    base_order = np.arange(len(axes))
    change_order = []
    for i in base_order:
        change_order.append(int((np.where(axes == i)-i) % len(axes)))
    
    value = np.transpose(t.value, axes=axes)
    have_grad = t.have_grad
    ops_name = '_transpose'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_transpose(grad: np.ndarray) -> np.ndarray:
            target_order = (change_order + base_order) % len(axes)
            return np.transpose(grad, axes=target_order)
            
        depends_on.append(T.Dependency(t, grad_fn_transpose, ops_name))

    else: 
        depends_on = []
        
    return T.Tensor(value, have_grad, depends_on)



def sinh(t: 'Tensor') -> 'Tensor':
    '''
        Hyperbolic Sinus calculation of tensor. Also it is calculate its 
        gradient of operation if tensor have_grad = True.
        Sinus in radian.
    '''
    value = np.sinh(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_sinh'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_sinh(grad: np.ndarray) -> np.ndarray:
            grad = np.cosh(t._value, dtype=np.float32) * grad.astype(np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_sinh, ops_name))

    else: 
        depends_on = []

    return T.Tensor(value, have_grad, depends_on)



def cosh(t: 'Tensor') -> 'Tensor':
    '''
        Hyperbolic Cosinus calculation of tensor. Also it is calculate its 
        gradient of operation if tensor have_grad = True.

        Cosinus in radian.
    '''
    value = np.cosh(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_cosh'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_cosh(grad: np.ndarray) -> np.ndarray:
            grad = np.sinh(t._value, dtype=np.float32) * grad.astype(np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_cosh, ops_name))

    else: 
        depends_on = []
    
    return T.Tensor(value, have_grad, depends_on)



def abs(t: 'Tensor') -> 'Tensor':
    value = np.absolute(t._value, dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_abs'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_abs(grad: np.ndarray) -> np.ndarray:
            grad = np.sign(t._value, dtype=np.float32) * grad.astype(np.float32)
            return grad

        depends_on.append(T.Dependency(t, grad_fn_abs, ops_name))
    
    return T.Tensor(value, have_grad, depends_on)



def dropout(t: 'Tensor', p: float) -> 'Tensor':
    """ 
        https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network
    """
    dropout_mask = np.random.binomial(1, 1.-p, size=t.shape)
    value = np.multiply(t._value, dropout_mask * (1./(1.-p)), dtype=np.float32)
    have_grad = t.have_grad
    ops_name = '_dropout'

    depends_on: List[Dependency] = []

    if have_grad:
        def grad_fn_dropout(grad: np.ndarray) -> np.ndarray:
            grad = grad.astype(np.float32) * dropout_mask * (1./(1.-p))
            return grad

        depends_on.append(T.Dependency(t, grad_fn_dropout, ops_name))
    
    return T.Tensor(value, have_grad, depends_on)



def append(t1: 'Tensor', t2: 'Tensor', axis=None) -> 'Tensor':
    t1_shape = t1.shape
    t2_shape = t2.shape
    value = np.append(t1.value, t2.value, axis)
    have_grad = t1.have_grad or t2.have_grad
    ops_name = '_append'

    depends_on: List[Dependency] = []

    if t1.have_grad:
        dim = np.arange(t1.value.ndim) # dimension
        ind = [] 
        [ind.append(slice(0,t1_shape[d])) for d in dim] # slice index
        def grad_fn_append1(grad: np.ndarray) -> np.ndarray:
            return grad[tuple(ind)]
        
        depends_on.append(T.Dependency(t1, grad_fn_append1, ops_name))

    if t2.have_grad:
        dim = np.arange(t2.value.ndim) # dimension
        ind2 = []
        [ind2.append(slice(-t2_shape[d]+value.shape[d],value.shape[d] )) for d in dim] #slice index
        def grad_fn_append2(grad: np.ndarray) -> np.ndarray:
            return grad[tuple(ind2)]
        
        depends_on.append(T.Dependency(t2, grad_fn_append2, ops_name))

    return T.Tensor(value, have_grad, depends_on)