import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_function(x, y):
    a, b = 1, 100
    z = (a - x)**2 + b*(y - x**2)**2
    return z

if __name__ == '__main__':

    x_min, x_max = -10, 10
    x_num = x_max + 1 - x_min
    x = np.linspace(x_min, x_max, num=x_num)

    y_min, y_max = -100, 100
    y_num = y_max + 1 - y_min
    y = np.linspace(y_min, y_max, num=y_num)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z,
                    rstride=1,
                    cstride=1,
                    cmap='winter',
                    edgecolor='none')

    ax.scatter(2,3,4, color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
