import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot

iter = np.array([[10], [100], [1000], [3000]])
n = 1
def grad_func_analytical(x):
    x1, x2, x3 = x
    return np.array([2 * (x1 - 5), 2 * (x2 - 9), -2 * x3 * np.exp(x3 ** 2)])


def grad_func_numerical(x, func):
    h = 1e-6
    g1 = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_plus[i] += h
        x_minus = np.copy(x)
        x_minus[i] -= h
        g1[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    h = h / 2
    g2 = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_plus[i] += h
        x_minus = np.copy(x)
        x_minus[i] -= h
        g2[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    g = richardson_extrapolation(g1, g2)
    return g


def richardson_extrapolation(g1, g2):
    p = 2
    alpha = (np.power(p,p) * g2 - g1) / (np.power(p,p) - 1)
    beta = np.power(p,p) - 1
    return (alpha * g2 - g1) / beta

def func(x):
    x1, x2, x3 = x
    return (x1 - 5) ** 2 + (x2 - 9) ** 2 + (1 - np.exp(x3 ** 2))

def adam(func, x_init, grad_func, max_iter, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha = 0.1 ):
    m = np.zeros_like(x_init)
    v = np.zeros_like(x_init)
    x = x_init
    results = []
    for t in range(max_iter):
        # Calculate the gradient using either analytical or numerical method
        if grad_func == 'analytical':
            g = grad_func_analytical(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))
            x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            results.append(func(x))
        elif grad_func == 'numerical':
            g = grad_func_numerical(x, func)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))
            x = (-1*x - alpha * m_hat / (np.sqrt(v_hat) + epsilon))*-1
            results.append(func(x))
        else:
            raise ValueError("Invalid value for grad_func: must be 'analytical' or 'numerical'")

    return x, np.array(results)

fig = plt.figure()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12)
max_it = 0
for max_it in iter:   # Run Adam with analytical gradients
    x_init = np.array([0, 0, 0])
    x_analytical, results_analytical = adam(func, x_init, 'analytical', int(max_it))
    # Run Adam with numerical gradients
    x_numerical, results_numerical = adam(func, x_init, 'numerical', int(max_it))
    # Plot the results
    ax = fig.add_subplot(2, 2, n)
    n = n + 1
    ax.plot(results_analytical, label='Analytical gradients')
    ax.plot(results_numerical, label='Numerical gradients')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Function value')
    title = 'the iteration is : ' + str(max_it)
    ax.set_title(title)
    print('when iteration is : ',max_it)
    print('Analytical solution: x =', x_analytical, 'f(x) = ', func(x_analytical))
    print('numerıcal solution: x =', x_numerical, 'f(x) = ', func(x_numerical),'\n\n\n')
plt.show()
fig.savefig('odev3.png', dpi=100)
