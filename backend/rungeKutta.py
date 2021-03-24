
import matplotlib.pyplot as plt
import numpy as np


# 4th order Runge Kutta method given a function, initial y value, the interval for x and step size h
def RK4thOrder(f, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0])/h)

    x = x_range[0]
    y = yinit

    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = h * f(x, y)

        k2 = h * f(x+h/2, y+k1/2)

        k3 = h * f(x+h/2, y+k2/2)

        k4 = h * f(x+h, y+k3)

        for j in range(m):
            y[j] = y[j] + (1/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    return [xsol, ysol]


# Function used for calculations, change this
def myFunc(x, y):
    dy = np.zeros((len(y)))
    dy[0] = np.sin(x)**(2*y[0])
    return dy


h = 0.1  # Step size
x = np.array([0.0, 20.0])  # Interval for x

# y(0) as array due to limitation for float datatype
yinit = np.array([1.0/10])

# Gives points from Runge Kutta method
[ts, ys] = RK4thOrder(f=myFunc, yinit=yinit, x_range=x, h=h)

# Calculates exact values for function, must be the same function as myfunc
# Maybe we don't need this
dt = int((x[-1]-x[0])/h)
t = [x[0]+i*h for i in range(dt+1)]
yexact = []
for i in range(dt+1):
    ye = (1.0/10)*np.exp(-2*t[i]) + t[i] * \
        np.exp(-2*t[i])  # change function here
    yexact.append(ye)

# Plotting graphs. Temporarily commented, research visualization with html
# Do we need to show this?


#plt.plot(ts, ys, 'r')
#plt.plot(t, yexact, 'b')
#plt.xlim(x[0], x[1])
#plt.legend(["4th Order RK", "Exact solution"], loc=1)
#plt.xlabel('x', fontsize=17)
#plt.ylabel('y', fontsize=17)
# plt.tight_layout()
# plt.show()
