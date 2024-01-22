import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return (x[0]-5)**2 + (x[1]-9)**2

def numerical_gradient(f, x, h=1e-4):
    """Numerically computes the gradient of a function at a point."""
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = h
        grad[i] = (f(x+dx) - f(x-dx)) / (2*h)
    return grad

def adam(f, x0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, h=1e-4):
    """Adam optimization algorithm for minimizing a function with numerical differentiation."""
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    while t < max_iter:
        t += 1
        g = numerical_gradient(f, x, h)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return x, f(x)






# Set initial guess
x0 = np.array([0, 0])

# Run Adam algorithm with numerical differentiation
x_min, f_min = adam(f, x0)

print(f"Minimum value: {f_min:.6f}")
print(f"Minimizer: {x_min}")

# Plot function and minimizer
x1 = np.linspace(-10, 20, 100)
x2 = np.linspace(-10, 20, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f([X1, X2])
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.contour(X1, X2, Z, levels=20)
if x_min is not None:
    ax.plot(x_min[0], x_min[1], 'ro', markersize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Adam optimizer path (2D)')


# Plot function and minimizer in 3D

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='coolwarm')
if x_min is not None:
    ax.plot([x_min[0]], [x_min[1]], [f_min], 'ro', markersize=10)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title('Surface plot of function')
plt.show()