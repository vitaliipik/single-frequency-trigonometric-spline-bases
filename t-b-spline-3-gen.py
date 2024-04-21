import numpy as np
import matplotlib.pyplot as plt


def trigonometric_bspline_basis(x, knots):
    """
    Calculate the trigonometric B-spline basis functions at a given point x.

    Args:
        x: Point at which to calculate the basis functions.
        knots: Array of knot points.

    Returns:
        Array of basis function values at x.
    """
    h = knots[-1] - knots[0]
    w = np.sin(h / 2) * np.sin(h) * np.sin(3 * (h) / 2)
    p = np.sin((x - knots) / 2)
    q = np.sin((knots - x) / 2)
    sizer = int(len(knots) / 2)
    basis = np.zeros(sizer)  # Initialize basis with proper length
    for i in range(sizer):
        if knots[i] <= x < knots[i + 1]:
            basis[i] = p[i] ** 3
        elif knots[i + 1] <= x < knots[i + 2]:
            basis[i] = p[i] * (p[i] * q[i + 2] + q[i + 3] * p[i + 1]) + q[i + 4] * p[i + 1] ** 2
        elif knots[i + 2] <= x < knots[i + 3]:
            basis[i] = q[i + 4] * (p[i + 1] * q[i + 3] + q[i + 4] * p[i + 2]) + p[i] * q[i + 3] ** 2
        elif knots[i + 3] <= x < knots[i + 4]:
            basis[i] = q[i + 4] ** 3

    return basis / w

def calculate_curve(control_points, knots, num_points=100):
    """
    Calculate the trigonometric B-spline curve points.

    Args:
        control_points: Array of control points (x, y).
        knots: Array of knot points.
        num_points: Number of points to calculate on the curve.

    Returns:
        Arrays of x and y coordinates of the curve points.
    """
    x_vals = np.linspace(knots[0], knots[-1], num_points)
    y_vals = np.zeros_like(x_vals)
    loh = []

    for i, x in enumerate(x_vals):
        result = np.zeros_like(control_points[0], dtype=float)
        basis_values = trigonometric_bspline_basis(x, knots)
        for index, f in enumerate(basis_values):
            result += f * control_points[index]
        loh.append(result)
    return np.array(loh)

# Example control points
control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])

# Example knots
knots = np.array([1, 1, 1, 1, 2, 2,2, 2])

# Calculate curve points
curve = calculate_curve(control_points, knots)

# Plot the curve
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.plot(curve[:, 0], curve[:, 1], label='Trigonometric B-spline Curve')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric B-spline Curve')
plt.legend()
plt.grid(True)
plt.show()
