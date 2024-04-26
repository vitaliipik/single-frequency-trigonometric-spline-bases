import re

import numpy as np


def duck():
    points_str = "(-0.2356, 0.3978), (-0.2044, 0.4178), (-0.1711, 0.4289), (-0.1467, 0.4733), (-0.1022, 0.4978), (-0.0533, 0.4933), (-0.0200, 0.4667), (0, 0.4444), (0.0089, 0.4111), (-0.0044, 0.3667), (-0.0333, 0.3311), (-0.0778, 0.2756), (-0.1067, 0.2400), (-0.1178, 0.2000), (-0.0889, 0.1778), (- 0.0511, 0.2156), (0.0156, 0.2533), (0.0844, 0.2778), (0.1467, 0.2956), (0.2111, 0.2911), (0.2556, 0.2644), (0.2578, 0.2222), (0.2267, 0.1911), (0.2667, 0.1800), (0.2622, 0.1467), (0.2222, 0.1111), (0.2467, 0.0933), (0.2267, 0.0556), (0.1800, 0.0289), (0.0200, 0.0244), (-0.1311, 0.0267), (-0.1711, 0.0711), (-0.2133, 0.1356), (-0.2133, 0.2067), (-0.1822, 0.2622), (-0.1311, 0.3178), (-0.1000, 0.3733), (- 0.1533, 0.3733), (-0.2178, 0.3689), (-0.2311, 0.3822), (-0.2356, 0.3978)"

    # Regular expression to extract points
    pattern = r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)"

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, points_str)

    # Convert matches to tuple of floats
    control_points = np.array([[float(x), float(y)] for x, y in matches])
    return control_points


def butterfly():
    # Define the range of theta
    theta = np.linspace(0, 2 * np.pi, 60)

    # Calculate r for each theta
    r = (np.sin(theta) + np.sin(3.5 * theta) ** 3) / 1000

    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    control_points = np.array([[x_i, y_i] for x_i, y_i in zip(x, y)])
    return control_points


def spiral(points=60):
    t = np.linspace(-2 * np.pi, 2 * np.pi, points)

    # Calculate x, y, and z for each t
    x = np.sin(3 * t) * np.cos(t)
    y = np.sin(3 * t) * np.sin(t)
    z = t
    control_points = np.array([[x_i, y_i, z_i] for x_i, y_i, z_i in zip(x, y, z)])
    return control_points


def spiral_2():
    t = np.linspace(0 + 1e-10, 4 * np.pi - 1e-10, 100)

    # Calculate x, y, and z for each t
    x = 2 * np.cos(t) - np.cos(2 * t)
    y = 2 * np.sin(t) - np.sin(2 * t)
    z = np.sqrt(8) * np.cos(2 / t)
    control_points = np.array([[x_i, y_i, z_i] for x_i, y_i, z_i in zip(x, y, z)])
    return control_points


def gerb():
    points = [
        (0, 9), (1, 8), (0.5, 2), (2, -1), (3, 0), (2, 1), (2.5, 5.5), (5, 8), (5, -4), (2, -4), (1, -6), (0, 9),
        (0, -7), (-1, -6), (-2, -4), (-5, -4), (-5, 8), (-2.5, 5.5), (-3, 0), (-2, -1), (-2, 1), (-1, 8), (0, 9),
        (4, -0.5), (4, -3), (2.5, -3), (2, -2), (4, -0.5), (-4, -0.5), (-4, -3), (-4, -0.5),
        (4, 0), (4, 5.5), (3.5, 5), (3, 1), (4, 0), (-4, 0), (-4, 5.5), (-3.5, 5), (-3, 1), (-4, 0),

        (1, -2), (1.5, -3), (0.5, -3), (1, -2), (-1, -2), (-1.5, -3), (-0.5, -3), (-1, -2),
        (-0.5, -4), (-1, -4), (-0.5, -5.5), (-0.5, -4), (0.5, -4), (1, -4), (0.5, -5.5), (0.5, -4),
        (0, 1), (-1, -1), (0, -2), (1, -1), (0, 1)
    ]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    # cs = CubicSpline(x, y, bc_type='periodic')

    # Generate x values for plotting
    x_interp = np.linspace(min(x), max(x), 100)
    y_interp = np.linspace(min(y), max(y), 100)
    # return np.array([[point[0], point[1]] for point in points])
    return np.array([[x_i, y_i] for x_i, y_i in zip(x_interp, y_interp)])