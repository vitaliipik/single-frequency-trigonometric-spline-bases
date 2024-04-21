import numpy as np


def trigonometric_bernstein_like_basis_functions(u, alpha, beta):
    """
    Calculate the trigonometric Bernstein-like basis functions
    """
    T0 = np.ones_like(u)
    T1 = np.sin(u) ** 2
    T2 = (1 - np.sin(u)) ** alpha
    T3 = (1 - np.cos(u)) ** beta
    return T0, T1, T2, T3


def compute_coefficients(alpha, beta, h, hi_minus_1, hi_plus_1):
    """
    Compute the coefficients based on the given expressions
    """
    gamma = (alpha - 1) * hi_minus_1 + (beta - 1) * hi_plus_1
    lamda = alpha * hi_minus_1 + beta * hi_plus_1
    phi = (alpha * gamma * hi_minus_1) / (2 * hi_minus_1 ** 2) + (lamda * beta * hi_plus_1) / (2 * hi_plus_1 ** 2)
    phi_prev = (beta * gamma * hi_minus_1) / (2 * hi_minus_1 ** 2) + (lamda * alpha * hi_minus_1) / (
                2 * hi_minus_1 ** 2)

    ai = (2 * alpha * beta * (beta - 1) * gamma * hi_minus_1 ** 2) / (phi_prev)
    di = (2 * alpha * alpha * beta * beta * gamma * lamda * h ** 2) / (phi)

    bi0 = (alpha * hi_minus_1) / lamda * phi * ai + (beta - 1) * hi_minus_1 / lamda * phi_prev * di
    ci0 = di
    bi1 = phi * ai
    ci1 = (lamda - alpha * hi_minus_1) / lamda * di
    bi2 = phi_prev * di / (beta * hi_plus_1)
    ci2 = phi * di / (alpha * hi_minus_1)
    bi3 = ai
    ci3 = (alpha * hi_minus_1) / lamda * phi * ai + (beta * hi_plus_1) / lamda * phi_prev * di

    return ai, bi0, bi1, bi2, bi3, ci0, ci1, ci2, ci3, di


def trigonometric_b_spline_like_basis_function(u, alpha, beta, knots):
    """
    Calculate the trigonometric B-spline-like basis function
    """
    n = len(knots) - 4
    result = np.zeros_like(u)

    for i in range(n):
        hi_minus_1 = knots[i + 1] - knots[i]
        hi = knots[i + 2] - knots[i + 1]
        hi_plus_1 = knots[i + 3] - knots[i + 2]

        ti = np.pi * (u - knots[i]) / (2 * hi)
        ti_plus_1 = np.pi * (u - knots[i + 1]) / (2 * hi_plus_1)
        ti_plus_2 = np.pi * (u - knots[i + 2]) / (2 * hi_plus_1)
        ti_plus_3 = np.pi * (u - knots[i + 3]) / (2 * hi_plus_1)

        T0, T1, T2, T3 = trigonometric_bernstein_like_basis_functions(ti, alpha[i], beta[i])
        T0_plus_1, T1_plus_1, T2_plus_1, T3_plus_1 = trigonometric_bernstein_like_basis_functions(ti_plus_1,
                                                                                                  alpha[i + 1],
                                                                                                  beta[i + 1])
        T0_plus_2, T1_plus_2, T2_plus_2, T3_plus_2 = trigonometric_bernstein_like_basis_functions(ti_plus_2,
                                                                                                  alpha[i + 2],
                                                                                                  beta[i + 2])
        T0_plus_3, T1_plus_3, T2_plus_3, T3_plus_3 = trigonometric_bernstein_like_basis_functions(ti_plus_3,
                                                                                                  alpha[i + 3],
                                                                                                  beta[i + 3])

        ai, bi0, bi1, bi2, bi3, ci0, ci1, ci2, ci3, di = compute_coefficients(alpha[i + 1], beta[i + 1], hi, hi_minus_1,
                                                                              hi_plus_1)

        result += (u >= knots[i]) * (u < knots[i + 1]) * (di * T3)
        result += (u >= knots[i + 1]) * (u < knots[i + 2]) * (
                    ci0 * T0_plus_1 + ci1 * T1_plus_1 + ci2 * T2_plus_1 + ci3 * T3_plus_1)
        result += (u >= knots[i + 2]) * (u < knots[i + 3]) * (
                    bi0 * T0_plus_2 + bi1 * T1_plus_2 + bi2 * T2_plus_2 + bi3 * T3_plus_2)
        result += (u >= knots[i + 3]) * (u < knots[i + 4]) * (ai * T0_plus_3)

    return result


# Example usage:
knots = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Example knot vector
alpha = np.array([3, 3, 3])  # Example alpha values
beta = np.array([3, 3, 3])  # Example beta values

# Generate points along the curve
num_points = 100
curve_points = trigonometric_b_spline_like_basis_function(np.linspace(0, 7, num_points), alpha, beta, knots)

# Plot the curve
import matplotlib.pyplot as plt

plt.plot(np.linspace(0, 7, num_points), curve_points, 'b-', label='Trigonometric B-spline-like Basis Function')
plt.legend()
plt.xlabel('u')
plt.ylabel('B(u)')
plt.title('Trigonometric B-spline-like Basis Function')
plt.grid(True)
plt.show()
