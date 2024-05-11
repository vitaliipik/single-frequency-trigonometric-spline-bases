import re
import yfinance as yf

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

# if "__main__" == __name__:
    # # Example usage:
    # control_points = np.array([[0, 0], [0,1], [3, 4], [6, 0], [7, 4]])
    # degree =2
    # knots = np.array([0,0,0, 1, 2, 3, 4, 3, 3,3])  # Example knot vector
    # knots = clamped_knot_vector(6,degree)
    # knots = np.array([0, 0,  0,   1,2, 3,  3, 3 ])
    # knots = np.array([0, 0,  0,   1,2, 3,  4,4 ,4])
    #
    # points_str = "(-0.2356, 0.3978), (-0.2044, 0.4178), (-0.1711, 0.4289), (-0.1467, 0.4733), (-0.1022, 0.4978), (-0.0533, 0.4933), (-0.0200, 0.4667), (0, 0.4444), (0.0089, 0.4111), (-0.0044, 0.3667), (-0.0333, 0.3311), (-0.0778, 0.2756), (-0.1067, 0.2400), (-0.1178, 0.2000), (-0.0889, 0.1778), (- 0.0511, 0.2156), (0.0156, 0.2533), (0.0844, 0.2778), (0.1467, 0.2956), (0.2111, 0.2911), (0.2556, 0.2644), (0.2578, 0.2222), (0.2267, 0.1911), (0.2667, 0.1800), (0.2622, 0.1467), (0.2222, 0.1111), (0.2467, 0.0933), (0.2267, 0.0556), (0.1800, 0.0289), (0.0200, 0.0244), (-0.1311, 0.0267), (-0.1711, 0.0711), (-0.2133, 0.1356), (-0.2133, 0.2067), (-0.1822, 0.2622), (-0.1311, 0.3178), (-0.1000, 0.3733), (- 0.1533, 0.3733), (-0.2178, 0.3689), (-0.2311, 0.3822), (-0.2356, 0.3978)"
    #
    # # Regular expression to extract points
    # pattern = r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"
    #
    # # Find all matches of the pattern in the input string
    # matches = re.findall(pattern, points_str)
    #
    # # Convert matches to tuple of floats
    # control_points = np.array([[float(x), float(y)] for x, y in matches])
    #
    # data = yf.download("AAPL", period="1y", interval="1d")
    # x = range(len(data))
    # # x = data.index
    # # y = data.iloc[:, -1].astype('float64')
    # y = data.iloc[:, 2].astype('float64')
    # control_points = np.array([[x_i, y_i] for x_i, y_i in zip(x, y)])
    #
    # num_knots = len(control_points) + degree + 1
    # knots = np.linspace(0, 1, num_knots)
    # # knots = clamped_knot_vector(6,degree)
    #
    # # Generate points along the curve
    # num_points = 1000
    # curve_points = [b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 3, num_points)]
    # curve_points=curve_points[:332]
    # # Plot the curve
    # import matplotlib.pyplot as plt
    #
    #
    # curve_points = np.array(curve_points)
    # plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
    # plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
    #
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('B-spline Curve')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()
