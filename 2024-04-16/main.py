# Generate Linear Data with uncertainty e
import numpy as np
import matplotlib.pyplot as plt
import random
data = []
# randomly intialize a slope m
m = 6
# Generate x points
for x in range(100):
    e = random.randint(0,150) # Use random pertubation from 0 - 150
    data.append(m * x + e) # append value of each point

x = np.arange(0,100,1) # store x-axis vals ; feature x
y = np.array(data) # y-axis function vals ; function f(x)
plt.scatter(x,y)

# Derivative of MSE wrt. b1
def dE_b1(x, y, b_1, b_o):
    s = 0
    n = len(x)
    for ele in range(n):
        s += x[ele] * y[ele] - b_1 * x[ele] ** 2 - b_o * x[ele]

    return -2 / n * s


# Deriative of MSE wrt. b0
def dE_bo(x, y, b_1, b_o):
    s = 0
    n = len(x)
    for ele in range(n):
        s += (y[ele] - b_1 * x[ele] - b_o)

    return -2 / n * s


def Gradient_Descent(x, y, a=0.01, epochs=25):
    # finding start point for gradient descent
    b_o = random.randint(0, 100) / 10
    b_1 = random.randint(0, 20) / 10

    # value storing
    loss_ = []
    form_ = []

    # equation: epoch = # of times
    for i in range(epochs):

        # gradient descent equation - attempting to
        b_1 = b_1 - a * dE_b1(x, y, b_1, b_o)
        b_o = b_o - a * dE_bo(x, y, b_1, b_o)

        # print(b_1)
        loss = abs(y - (b_1 * x + b_o)).sum()
        loss_.append(loss)
        form_.append([b_1, b_o])

    return b_1, b_o, loss_, form_


b_1, b_o, loss_, form_ = Gradient_Descent(x, y, a=0.0001, epochs=10)
plt.plot(b_1 * x + b_o, c='g', linewidth=3)
plt.scatter(x, y)

#%%

#%%
