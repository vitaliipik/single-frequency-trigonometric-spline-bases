import numpy as np


def basis_function(i, k, t, knots):
    """
    Calculate the i-th B-spline basis function of order k at parameter t
    """
    if k == 1:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]

    coeff1 = 0.0 if denom1 == 0.0 else (t - knots[i]) / denom1
    coeff2 = 0.0 if denom2 == 0.0 else (knots[i + k] - t) / denom2

    return coeff1 * basis_function(i, k - 1, t, knots) + coeff2 * basis_function(i + 1, k - 1, t, knots)


def b_spline_curve(control_points, degree, knots, t):
    """
    Calculate the B-spline curve at parameter t
    """
    n = len(control_points) - 1  # number of control points
    result = np.zeros_like(control_points[0], dtype=float)  # Initialize result with float dtype

    for i in range(n + 1):
        basis = basis_function(i, degree + 1, t, knots)
        result += control_points[i] * basis

    return result


# Example usage:
control_points = np.array([[0, 0], [1, 3], [2, 4], [3, 4], [4, 0],[6, 2]])
degree = 2

knots = np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3,5,5])  # Example knot vector

def open_knot_vector(num_control_points, degree):
    """
    Generate an open knot vector for a B-spline curve
    """
    num_knots = num_control_points + degree + 1
    knots = np.zeros(num_knots)
    knots[:degree + 1] = 0
    knots[-(degree + 1):] = 1
    inner_knots = np.linspace(0, 1, num_control_points - degree - 1, endpoint=False)
    knots[degree + 1:num_control_points] = inner_knots
    return knots
def clamped_knot_vector(num_control_points, degree):
    """
    Generate a clamped knot vector for a B-spline curve
    """
    num_knots = num_control_points + degree + 1
    knots = np.zeros(num_knots)
    knots[:degree + 1] = 0
    knots[-(degree + 1):] = 1
    inner_knots = np.linspace(0, 1, num_control_points - degree - 1)
    knots[degree + 1:num_control_points] = inner_knots
    return knots

# Example usage:
control_points = np.array([[0, 0], [0,1], [3, 4], [6, 0], [7, 4]])
degree =2
knots = np.array([0,0,0, 1, 2, 3, 4, 3, 3,3])  # Example knot vector
knots = clamped_knot_vector(6,degree)
knots = np.array([0, 0,  0,   1,2, 3,  3, 3 ])
knots = np.array([0, 0,  0,   1,2, 3,  4,4 ,4])

# knots = clamped_knot_vector(6,degree)

# Generate points along the curve
num_points = 100
curve_points = [b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 3, num_points)]

# Plot the curve
import matplotlib.pyplot as plt

curve_points = np.array(curve_points)
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-spline Curve')
plt.grid(True)
plt.axis('equal')
plt.show()
