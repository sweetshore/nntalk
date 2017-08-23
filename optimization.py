from matplotlib import pyplot as plt

def profit_fun(x):
    """Sample profit function for cat house business"""
    costs = 20 + 3 * x
    price = 8 - 0.02 * x
    profit = price * x - costs
    return profit

def grid_search(fun, x_min, x_max, num_steps):
    x_step = (x_max - x_min) / num_steps
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


def gradient(fun, x, eps):
    """Returns function grow rate (aka gradient)"""
    f_plus = fun(x + eps)
    f_minus = fun(x - eps)
    return (f_plus - f_minus) / (2 * eps)

def gradient_search(fun, x_start, rate, num_steps):
    steps = []
    x = x_start
    for n in range(num_steps):
        grad = gradient(fun, x, 0.01)
        x += rate * grad
        steps.append((x, fun(x)))
    return x, fun(x), steps


def plot_search_steps(fun, x_min, x_max, steps):
    # plot smooth curve in range
    plot_points = 1000
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


if __name__ == "__main__":
    #best_x, best_value, steps = grid_search(profit_fun, 0, 200, 30)
    #best_x, best_value, steps = constant_step_search(profit_fun, 100, 1, 30)
    best_x, best_value, steps = gradient_search(profit_fun, 100, 10, 30)

    print("solution", best_x, best_value)
    plot_search_steps(profit_fun, 0, 200, steps)
