import numpy as np
import matplotlib.pyplot as plt

# Define parameters
alpha = 1.0
beta = 1.0
xi = 1.0

# Define knot vector
N = np.array([0, 1, 2, 3, 4])  # Adjust as needed

# Define control points
S = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])  # Adjust as needed


# Define function to evaluate spline curve
def Fj(n, j):
    q = np.diff(N)
    qj = N[j + 1] - N[j]
    alpha_j = 1 / (qj + q[j])
    beta_j = 1 / (q[j - 1] + q[j - 2])
    gamma_j = 2 / (q[j - 2] + q[j - 1] + qj)

    psi_j = 6 * xi ** 2 * alpha_j * q[j - 1] ** 2 - 3 * xi * q[j] + 10 * beta_j + q[j + 1] ** 2
    phi_j = (psi_j * (q[j + 1] + 1)) / (2 * (beta_j * q[j - 1] ** 3 + alpha_j * q[j + 1] ** 2))
    a = (2 * alpha_j * beta_j * q[j] ** 3) / psi_j
    tj = np.pi / 2 * (n - N[j]) / qj

    if j == 0:
        return (1 - np.sin(tj)) ** 2 * (xi ** 2 - xi + 1 - np.sin(tj))
    elif j == len(N) - 2:
        return alpha * (1 - np.cos(tj)) ** 2 * (xi ** 2 - xi + 1 - np.cos(tj))
    else:
        sum_cj = 0
        for k in range(4):
            if k == 0:
                cj = (1 - (xi ** 2 - 3 * xi - 1) * a) * gamma_j * (q[j - 1] - q[j])
            elif k == 1:
                cj = (3 * gamma_j * q[j - 1] ** 2 * alpha_j) / (2 * phi_j)
            elif k == 2:
                cj = ((1 - 2 * xi) * beta_j * q[j - 1] * sum_cj) / phi_j
            elif k == 3:
                cj = 4 * xi * (q[j + 1] - q[j]) * gamma_j
            sum_cj += cj
        return sum_cj


# Evaluate the spline curve
x_values = np.linspace(0, 4, 100)
y_values = np.array([Fj(x, j) * S[j] for j in range(len(N) - 2) for x in x_values])


# def spline(t,s):
#     n = len(s) - 1  # number of control points
#     result = np.zeros_like(s[0], dtype=float)  # Initialize result with float dtype
#
#     for i in range(n + 1):
#         basis = Fj(t,i)
#         result += s[i] * basis
#
#     return result
#
#  # Initialize result with float dtype
#
#
# num_points = 100
# curve_points = [spline(t,S) for t in np.linspace(0, 3, num_points)]

curve_points = np.array(y_values)
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
plt.plot(S[:, 0], S[:, 1], 'ro-', label='Control Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric B-spline Curve')
plt.legend()
plt.grid(True)
plt.show()
