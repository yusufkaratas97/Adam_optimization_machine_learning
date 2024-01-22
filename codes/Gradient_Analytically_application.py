import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the function to be optimized
def f(x, y):
    return (x - 5) ** 2 + (y - 9) ** 2


# Define the Adam optimizer
def adam_optimizer(grad, m, v, t, beta1=0.9, beta2=0.999, lr=0.01, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    delta = lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return delta, m, v


# Initialize the starting point
x = np.array([0, 0])

# Initialize the momentum and velocity vectors for Adam
m = np.zeros(2)
v = np.zeros(2)

# Store the history of the path taken by the optimizer
path = [x]

# Iterate the optimizer for 50 steps
for i in range(50):
    # Compute the gradient of the function at the current point
    grad = np.array([2 * (x[0] - 5), 2 * (x[1] - 9)])
    # Update the model with the gradient using the Adam optimizer
    delta, m, v = adam_optimizer(grad, m, v, i + 1)
    x = x - delta

    # Store the new point in the path
    path.append(x)

# Print the minimum value of the function and the coordinates of the minimum point
min_value = f(x[0], x[1])
min_point = x
print(path)
print("Minimum value: ", min_value)
print("Minimum point: ", min_point)

# Plot the function and the path taken by the optimizer in 2D
x_range = np.arange(-10, 20, 0.1)
y_range = np.arange(-10, 20, 0.1)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.contour(X, Y, Z, levels=20)
ax.plot([p[0] for p in path], [p[1] for p in path], 'ro-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Adam optimizer path (2D)')

# Plot the function and the path taken by the optimizer in 3D
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.plot([p[0] for p in path], [p[1] for p in path], [f(p[0], p[1]) for p in path], 'ro-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Adam optimizer path (3D)')
plt.show()