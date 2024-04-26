import re
import subprocess

import numpy as np
from plotting import plot_basis
from preset_figure import spiral

INPUT_FILE_CONTROL_POINTS = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\control_points.txt"
INPUT_FILE_KNOTS = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\knots.txt"
OUTPUT_FILE = r"\\wsl.localhost\Ubuntu\home\vitalii\cuda-mch\result.txt"

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

def write_points(points, filename):
    with open(filename, 'w') as file:
        file.write(str(len(points)))

        for point in points:
            if len(point) == 3:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                file.write(f"{point[0]} {point[1]}\n")

def write_knots(knots, filename):
    with open(filename, 'w') as file:
        file.write(str(len(knots)))
        for point in zip(knots):
            file.write(f"{point}\n")

def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            points.append((float(values[0]), float(values[1])))

    return points

def main():


    control_points=spiral()


    degree =2

    # Generate points along the curve
    num_points = 1000

    num_knots = len(control_points) + degree + 1
    knots = np.linspace(0, 1, num_knots-4)

    knots =np.concatenate([[0]*2,knots,[1]*2])

    write_points(control_points, INPUT_FILE_CONTROL_POINTS)
    write_knots(knots, INPUT_FILE_KNOTS)





    # command = ["mpiexec", "-np", str(num_procs), "./cpp/hello-world", str(N)]
    # result = subprocess.run(command, capture_output=True, text=True)

    command = ["wsl","./cuda-mch/test",str(degree),str(num_points),INPUT_FILE_CONTROL_POINTS,INPUT_FILE_KNOTS,OUTPUT_FILE]
    subprocess.run(command, capture_output=True, text=True)

    # curve_points = np.array([b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, 3, num_points)])

    curve_points=read_points(OUTPUT_FILE)

    curve_points=curve_points[:333]

    plot_basis(control_points,curve_points)

if __name__ == '__main__':
    main()