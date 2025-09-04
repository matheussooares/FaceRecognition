from numpy import array, exp, where, clip

def step(x:array) -> array:
    return where(x < 0, 0, 1)

def sigmoid(x:array) -> array:
    x = clip(x, -5000, 5000)
    return 1 / (1 + exp(-x))

def tanh(x:array) -> array:
    return (1 - exp(-x)) / (1+ exp(-x))


def d_sigmoid(Y):
    return Y*(1 - Y)

def d_tanh(Y):
    return 1/2 *(1- Y**2)

def d_step(Y):
    return 0 
