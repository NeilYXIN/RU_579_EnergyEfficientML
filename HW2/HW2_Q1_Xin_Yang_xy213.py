import math

def x_times_y(x, y):
    print("x*y", round(x*y, 2))
    return x*y
def sin(x):
    print("sin", round(math.sin(x),2))
    return math.sin(x)

def cos(x):
    print("cos", round(math.cos(x),2))
    return math.cos(x)

def square(x):
    print("^2", round(x**2,2))
    return x**2

def plus_2(x):
    print("+2", round(x+2, 2))
    return x+2

def x_plus_y(x, y):
    print("x+y", round(x+y, 2))
    return(x+y)

def reciprocal(x):
    print("1/x", round(1/x, 2))
    return 1/x

def forward_prop(x1, w1, x2, w2):
    x1_times_w1 = x_times_y(x1, w1)
    sin_x1_times_w1 = sin(x1_times_w1)
    square_sin = square(sin_x1_times_w1)

    x2_times_w2 = x_times_y(x2, w2)
    cos_x2_times_w2 = cos(x2_times_w2)

    square_sin_plus_cos = x_plus_y(square_sin, cos_x2_times_w2)

    two_plus_square_sin_plus_cos = plus_2(square_sin_plus_cos)

    forward_res = reciprocal(two_plus_square_sin_plus_cos)
    print("Forward propagation", round(forward_res, 2))
    print(" ")
    return {"x1_times_w1": x1_times_w1, "sin_x1_times_w1":sin_x1_times_w1, "square_sin":square_sin, 
    "x2_times_w2":x2_times_w2, "cos_x2_times_w2":cos_x2_times_w2, 
    "square_sin_plus_cos":square_sin_plus_cos, "two_plus_square_sin_plus_cos":two_plus_square_sin_plus_cos,
    "forward_res": forward_res}

def dx_times_y_dx(x, y):
    return y

def dx_times_y_dy(x, y):
    return x

def dsinx(x):
    return math.cos(x)

def dcos(x):
    return -math.sin(x)

def dsquare(x):
    return 2*x

def dplus_2(x):
    return 1

def dreciprocal(x):
    return -1/(x**2)

def dx_plus_y_dx(x, y):
    return 1

def dx_plus_y_dy(x, y):
    return 1

def back_prop(dict_forward):
    x1_times_w1 = dict_forward["x1_times_w1"]
    sin_x1_times_w1 = dict_forward["sin_x1_times_w1"]
    square_sin = dict_forward["square_sin"]

    x2_times_w2 = dict_forward["x2_times_w2"]
    cos_x2_times_w2 = dict_forward["cos_x2_times_w2"]

    square_sin_plus_cos = dict_forward["square_sin_plus_cos"]
    two_plus_square_sin_plus_cos = dict_forward["two_plus_square_sin_plus_cos"]

    # forward_res = dict_forward["forward_res"]

    grad_forward_res = 1.00
    print(grad_forward_res)

    grad_two_plus_square_sin_plus_cos = grad_forward_res * dreciprocal(two_plus_square_sin_plus_cos)
    print(grad_two_plus_square_sin_plus_cos)

    grad_square_sin_plus_cos = grad_two_plus_square_sin_plus_cos * dplus_2(square_sin_plus_cos)
    print(grad_square_sin_plus_cos, "\n")

    grad_square_sin = grad_square_sin_plus_cos * dx_plus_y_dx(square_sin, cos_x2_times_w2)
    print(grad_square_sin)
    grad_sin_x1_times_w1 = grad_square_sin * dsquare(sin_x1_times_w1)
    print(grad_sin_x1_times_w1)
    grad_x1_times_w1 = grad_sin_x1_times_w1 * dsinx(x1_times_w1)
    print(grad_x1_times_w1, "\n")

    grad_x1 = grad_x1_times_w1 * dx_times_y_dx(x1, w1)
    print(grad_x1)
    grad_w1 = grad_x1_times_w1 * dx_times_y_dy(x1, w1)
    print(grad_w1, "\n")

    grad_cos_x2_times_w2 = grad_square_sin_plus_cos * dx_plus_y_dy(square_sin, cos_x2_times_w2)
    print(grad_cos_x2_times_w2)
    grad_x2_times_w2 = grad_cos_x2_times_w2 * dcos(x2_times_w2)
    print(grad_x2_times_w2, "\n")

    grad_x2 = grad_x2_times_w2 * dx_times_y_dx(x2, w2)
    print(grad_x2)
    grad_w2 = grad_x2_times_w2 * dx_times_y_dy(x2, w2)
    print(grad_w2, "\n")

    print("Gradients:")
    print("x1", round(grad_x1, 2))
    print("w1", round(grad_w1, 2))
    print("x2", round(grad_x2, 2))
    print("w2", round(grad_w2, 2))

x1 = 1.0
w1 = 1.0
x2 = 1.0
w2 = 1.0

dict_forward = forward_prop(x1, w1, x2, w2)
back_prop(dict_forward)