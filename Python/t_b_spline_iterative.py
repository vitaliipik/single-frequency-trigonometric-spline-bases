import re

import numpy as np
from scipy.interpolate import CubicSpline
from plotting import plot_basis
from preset_figure import spiral, duck, star


def basis_function(i, k, t, knots):
    """
    Calculate the i-th B-spline basis function of order k at parameter t
    """
    if k == 1:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]

    coeff1 = 0.0 if denom1 == 0.0 else np.sin((t - knots[i])/2) / np.sin(denom1/2)
    coeff2 = 0.0 if denom2 == 0.0 else np.sin((knots[i + k] - t)/2) / np.sin((knots[i + k] - knots[i+1])/2)

    return coeff1 * basis_function(i, k - 1, t, knots) + coeff2 * basis_function(i + 1, k - 1, t, knots)


def t_b_spline_curve(control_points, degree, knots, t):
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
degree = 5

knots = np.array([1, 1, 1, 0, 1, 2, 3, 3, 4, 4,4,4])  # Example knot vector

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
# control_points = np.array([[0, 0], [0,1], [3, 4], [6, 0], [7, 4]])
degree =2
# knots = np.array([0,0,0, 1, 2, 3, 4, 3, 3,3])  # Example knot vector
# knots = clamped_knot_vector(6,degree)
# # knots = np.array([0, 0,  0,   1,2, 3,  3, 3 ])
# knots = np.array([0, 0,  0,   1,2, 3,  4,4 ,4])
# knots = np.array([0, 0,  0,0,1,   2,3, 4, 5,5 ,5,5])
# # knots = np.array([0, 1,  2,   3,4, 5, 6,7,8,9,10])



control_points=star()

# degree =2
# knots = np.array([0,0,0, 1, 2, 3, 4, 3, 3,3])  # Example knot vector
# knots = clamped_knot_vector(6,degree)
# # knots = np.array([0, 0,  0,   1,2, 3,  3, 3 ])
# knots = np.array([0, 0,  0,   1,2, 3,  4,4 ,4])
# knots = np.array([0, 0,  0,0,1,   2,3, 4, 5,5 ,5,5])
knots = np.array([0, 1,  2,   3,4, 5, 6,7,8,9,10])
num_knots = len(control_points) + degree + 1
knots = np.linspace(0, 1, num_knots-4)
# knots = clamped_knot_vector(6,degree)
knots =np.concatenate([[0]*2,knots,[1]*2])

# knots[1:3]=0
# knots[-3:-1]=knots[-1]

# Generate points along the curve
num_points = 1000
curve_points = np.array([t_b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 3, num_points)])

curve_points=curve_points[:333]

plot_basis(control_points,curve_points)