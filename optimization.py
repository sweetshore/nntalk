from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

### 1-d optimization 

def profit_fun(x):
    """Sample profit function for cat house business"""
    costs = 20 + 3 * x
    price = 8 - 0.02 * x
    profit = price * x - costs
    return profit

def grid_search(fun, x_min, x_max, num_steps):
    x_step = (x_max - x_min) / (num_steps - 1)
    steps = []
    best_value = -1e100
    best_x = x_min
    for n in range(num_steps):
        x = x_min + x_step * n
        val = fun(x)
        if val > best_value:
            best_value = val
            best_x = x
        steps.append((x, val))
    return best_x, best_value, steps


def is_growing(fun, x, eps):
    """Returns True if function grows when x increases, False otherwise"""
    f_plus = fun(x + eps)
    f_minus = fun(x - eps)
    return f_plus > f_minus

def constant_step_search(fun, x_start, x_step, num_steps):
    steps = []
    x = x_start
    for n in range(num_steps):
        if is_growing(fun, x, 0.01):
            x += x_step
        else:
            x -= x_step
        steps.append((x, fun(x)))
    return x, fun(x), steps


def gradient_1d(fun, x, eps):
    """Returns function grow rate (aka gradient) for 1-variable functions"""
    return (fun(x + eps) - fun(x - eps)) / (2 * eps)

def gradient_search_1d(fun, x_start, rate, num_steps):
    """Search for 1-d function maximum by gradient descent"""
    steps = [] 
    x = x_start
    steps.append((x, fun(x))) # add x-start
    for n in range(num_steps):
        grad = gradient_1d(fun, x, 0.01)
        x += rate * grad
        steps.append((x, fun(x)))
    return x, fun(x), steps

### 2-d optimization 

def loss_fun(x):
    """Sample loss function for two variable problem""" 
    x1 = x[0]
    x2 = x[1]
    f = 5 * (x1 - 4)**2 + 3 * (x2 - 5)**2 
    #f = 20 * (x1 - 4)**2 + 3 * (x2 - 5)**2 
    return f

def gradient(fun, x, eps):
    """Search for function minimum by gradient descent"""
    grad = np.zeros(len(x)) # array with zeros [0 0 .. 0] same size as x
    for n in range(len(x)):
        x_plus = np.copy(x) # copy values of array, not reference to old array
        x_plus[n] += eps
        x_minus = np.copy(x)
        x_minus[n] -= eps
        grad[n] = (fun(x_plus) - fun(x_minus)) / (2 * eps)
    return grad

def gradient_search(fun, x_start, rate, num_steps):
    steps = []
    x = x_start
    steps.append((np.copy(x), fun(x)))
    for n in range(num_steps):
        grad = gradient(fun, x, 0.01)
        x -= rate * grad
        steps.append((np.copy(x), fun(x)))
    return x, fun(x), steps

### plot functions

def plot_search_steps(fun, x_min, x_max, steps):
    # plot smooth curve in range
    plot_points = 100
    x_step = (x_max - x_min) / plot_points
    x = []
    y = []
    for n in range(plot_points):
        curr_x = x_min + n * x_step
        x.append(curr_x)
        y.append(fun(curr_x))
    plt.plot(x, y, '-b')
    
    # plot search steps
    x = []
    y = []
    for step_x, step_y in steps:
        x.append(step_x)
        y.append(step_y)
    plt.plot(x, y, 'o-r')
            
    # display plot
    plt.show()

def plot_search_steps_3d(fun, x_min, x_max, steps = []):
    
    plot_points = 100
    x_step = (x_max - x_min) / plot_points

    x = np.arange(x_min, x_max, x_step)
    y = np.arange(x_min, x_max, x_step)
    x, y = np.meshgrid(x, y)
    z = [fun([x[n], y[n]]) for n in range(len(x))]

    steps_x = []
    steps_y = []
    steps_z = []
    for point, val in steps:
        steps_x.append(point[0])
        steps_y.append(point[1])
        steps_z.append(val)
            
    # display plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot search steps    
    ax.plot(steps_x, steps_y, zs=steps_z, color='r', linestyle='solid', marker='o')

    # plot the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == "__main__":
    
    # 1-d optimization
    best_x, best_value, steps = grid_search(profit_fun, 0, 200, 30)
    #best_x, best_value, steps = constant_step_search(profit_fun, 100, 1, 30)
    #best_x, best_value, steps = gradient_search_1d(profit_fun, 100, 10, 30)

    print("solution", best_x, best_value)
    plot_search_steps(profit_fun, 0, 200, steps)

    # 2-d optimization
    #x_start = np.array([9.0, 9.0])
    #rate = 0.05 #0.01
    #best_x, best_value, steps = gradient_search(loss_fun, x_start, rate, 30)
    #print("solution", best_x, best_value)
    #plot_search_steps_3d(loss_fun, 0.0, 10.0, steps)
