from preset_figure import spiral, duck
import numpy as np
from matplotlib import pyplot as plt

from plotting import plot_basis, line_plot
def write_points(points, filename):
    with open(filename, 'w') as file:
        file.write(str(len(points)) + "\n")

        for point in points:
            if len(point) == 3:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                file.write(f"{point[0]} {point[1]}\n")

def write_knots(knots, filename):
    with open(filename, 'w') as file:
        file.write(str(len(knots)) + "\n")
        for point in knots:
            file.write(f"{point}\n")


def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            points.append((float(values[0]), float(values[1])))

    return points


def main():
    control_points = duck()

    degree = 2
    alpha = np.pi / 4
    # alpha=1.33

    # Generate points along the curve
    num_points = 1000

    p = len(control_points)
    knots = [0, 0, 0]
    for i in range(3, p + 1):
        knots += [(i - 2) * alpha]
    knots += [(p - 1) * alpha] * 3
    # write_points(control_points, "/Users/admin/CLionProjects/single-frequency-trigonometric-spline-bases-/control_poins.txt")
    # write_knots(knots, "/Users/admin/CLionProjects/single-frequency-trigonometric-spline-bases-/knots.txt")
    curve_points = np.array(read_points("/Users/admin/CLionProjects/single-frequency-trigonometric-spline-bases-/res.txt"))
    curve_points = curve_points[~np.all(curve_points == 0, axis=1)]

    # curve_points=curve_points[:333]
    plt.show()
    plot_basis(control_points, curve_points)
    plt.show()
    space = 100

if __name__ == "__main__":
    main()
