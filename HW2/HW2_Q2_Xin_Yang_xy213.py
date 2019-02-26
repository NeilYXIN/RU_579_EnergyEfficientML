import math
import numpy as np

def W_times_x(W, x):
    print("W*x", np.matmul(W, x), "\n")
    return np.matmul(W, x)

def sigmoid(x):
    print("Sigmoid", math.e**x/(math.e**x + 1), "\n")
    return math.e**x/(math.e**x + 1)

def L2(x):
    squared = np.square(x)
    print("L2", np.sum(squared), "\n")
    return np.sum(squared)



def foward_prop(W, x):
    w_times_x = W_times_x(W, x)
    sigmoid_w_times_x = sigmoid(w_times_x)
    L2_sigmoid = L2(sigmoid_w_times_x)
    print("Forward propagation", L2_sigmoid, "\n")
    return {
        "w_times_x": w_times_x,
        "sigmoid_w_times_x": sigmoid_w_times_x,
        "L2_sigmoid": L2_sigmoid
    }

def dL2_dx(x):
    return 2*x

def dsigmoid(x):
    print("dsigmoid",(np.ones_like(x)-sigmoid(x))*sigmoid(x))
    return (np.ones_like(x)-sigmoid(x))*sigmoid(x)

def dW_timesx_dW(W, x):
    return x

def dW_timesx_dx(W, x):
    return W

def back_prop(foward_dict):
    w_times_x = foward_dict["w_times_x"]
    sigmoid_w_times_x = foward_dict["sigmoid_w_times_x"]
    L2_sigmoid = foward_dict["L2_sigmoid"]

    grad_L2 = 1.00
    grad_sigmoid = grad_L2 * dL2_dx(sigmoid_w_times_x)
    print("grad_sigmoid", grad_sigmoid)
    grad_w_times_x = grad_sigmoid * dsigmoid(w_times_x)
    print("grad_w_times_x", grad_w_times_x, "\n")

    grad_W = grad_w_times_x * np.transpose(dW_timesx_dW(W, x))
    grad_x = np.matmul(np.transpose(W), grad_w_times_x)

    print("Gradients:")
    print("W", grad_W)
    print("x", grad_x)



W = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]

x = [
    [1.0],
    [1.0],
    [1.0]
]

forward_dict = foward_prop(W, x)
back_prop(forward_dict)