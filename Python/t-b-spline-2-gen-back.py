import numpy as np


# Define basis functions l_i(t)
def l_functions(t, xi):
    l0 = (1 - np.sin(t)) ** 2 * (xi ** 2 - xi + 1 - np.sin(t))
    l1 = (1 + np.cos(t)) ** 2 * (xi ** 2 - xi + 1 + np.cos(t))
    l2 = (1 + np.sin(t)) ** 2 * (xi ** 2 - xi + 1 + np.sin(t))
    l3 = (1 - np.cos(t)) ** 2 * (xi ** 2 - xi + 1 - np.cos(t))
    return l0, l1, l2, l3


# Define basis functions F_j(n)
def F_functions(n, j, alpha, beta, gamma, phi, psi, xi, tilde_xi, tilde_zeta, q, control_points):
    # Calculate t_j
    t_j = np.pi / 2 * ((n - q[j]) / (q[j + 1] - q[j]))

    # Ensure t_j stays within the valid range for sine function
    # t_j = np.clip(t_j, -np.pi / 2, np.pi / 2)

    # Calculate alpha, beta, gamma, phi, psi for this step
    q_j = q[j + 1] - q[j]
    alpha_j = 1 / (q_j + q[j])
    beta_j = 1 / (q[j - 1] + q[j - 2])
    gamma_j = 2 / (q[j - 1] + q[j] + q[j + 1])
    phi_j = (psi[j - 2] * (q[j + 1] + 1)) / (2 * (beta[j - 1] * q[j - 1] ** 3 + alpha_j * q[j + 1] ** 2))
    psi_j = 6 * tilde_xi ** 2 * alpha[j - 1] * q[j - 1] ** 2 - 3 * tilde_xi * q_j + 10 * beta[j + 1] * q[j + 1] ** 2
    daun = (alpha_j + 2 * q[j]) / (2 * beta_j * psi_j)
    # Define F_j(n) based on n's interval
    if j ==3:
        F = daun*control_points[j] * l_functions(t_j, xi)[0]
    elif j ==2:
        b0 = (1 - (xi ** 2 - 3 * xi - 1) * control_points[j + 1]) * gamma[j + 1] * (q[j - 1] - q_j)
        b1 = (3 * gamma[j + 1] * q[j - 1] ** 2 * alpha_j) / (2 * phi[j - 3])
        b2 = ((1 - 2 * tilde_zeta) * beta[j - 1] * q[j - 1] * b0) / phi[j + 1]
        b3 = 4 * tilde_xi * (q[j + 1] - q_j) * gamma_j
        F = b0 * l_functions(t_j, xi)[0] + b1 * l_functions(t_j, xi)[1] + b2 * l_functions(t_j, xi)[2] + b3 * \
            l_functions(t_j, xi)[3]
    elif j == 1:
        c0 = (q[j + 2] - q[j + 1]) * alpha[j + 1] * F_functions(n, j - 1, alpha, beta, gamma, phi, psi, xi, tilde_xi,
                                                                tilde_zeta, q, control_points)
        c3 = -3 * beta[j - 1] * (q[j + 2] - q[j - 1]) * (gamma[j + 2] * q[j + 1] + alpha[j + 1] * q_j) * alpha[j + 2]
        c2 = 4 * phi[j - 1] / (3 * gamma[j + 1] * q[j + 1])
        c1 = (12 * gamma_j * alpha[j + 1] * q_j ** 3 * psi[j - 1] * c0) / phi[j + 1] ** 2
        F = c0 * l_functions(t_j, xi)[0] + c1 * l_functions(t_j, xi)[1] + c2 * l_functions(t_j, xi)[2] + c3 * \
            l_functions(t_j, xi)[3]
    elif j == 0:
        F = alpha_j * l_functions(t_j, xi)[3]*control_points[j]
    else:
        F = np.zeros_like(control_points[0])

    return F


# Calculate spline curve G(n)
def cubic_trigonometric_bspline_curve(n, control_points, alpha, beta, gamma, phi, psi, xi, tilde_xi, tilde_zeta, q):
    G = np.zeros_like(control_points[0],dtype=float)
    for j in range(4):
        G += F_functions(n, j, alpha, beta, gamma, phi, psi, xi, tilde_xi, tilde_zeta, q, control_points)
    return G


# Example usage
if __name__ == "__main__":
    # Define control points
    control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0], [4, 2]])

    # Define parameters (update them as needed)
    alpha = np.ones(5)
    beta = np.ones(5)
    gamma = np.ones(5)
    phi = np.ones(5)
    psi = np.ones(5)
    xi = 0.5
    tilde_xi = 0.5
    tilde_zeta = 0.5
    q = np.array([1, 2, 3, 4, 5])  # Assuming uniform knot vector

    # Compute the spline curve for n in the range [0, 10]
    n_values = np.linspace(0, 3, 100)
    spline_curve = np.array(
        [cubic_trigonometric_bspline_curve(n, control_points, alpha, beta, gamma, phi, psi, xi, tilde_xi, tilde_zeta, q)
         for n in n_values])

    # Plot the spline curve
    import matplotlib.pyplot as plt

    plt.plot(spline_curve[:, 0], spline_curve[:, 1], label='Cubic Trigonometric B-spline Curve')
    plt.scatter(control_points[:, 0]*17, control_points[:, 1]*17, color='red', label='Control Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Trigonometric B-spline Curve')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
