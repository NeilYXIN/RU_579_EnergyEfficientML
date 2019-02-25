import math

def sigmoid(x):
    return math.exp(x)/(math.exp(x) + 1)

print(sigmoid(3))
print(0.95**2 + 0.95**2 + 0.95**2)