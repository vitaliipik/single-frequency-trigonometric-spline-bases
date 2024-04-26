import matplotlib.pyplot as plt
import numpy as np


def plot_basis(control_points, curve_points, is_control=True):
    if len(curve_points[0]) == 3:

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the curve
        if is_control:
            ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro-',
                    label='Three-dimensional three-leaf rose curve')
        ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], 'b-',
                label='Three-dimensional three-leaf rose curve')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Three-dimensional three-leaf rose curve')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()
    else:

        curve_points = np.array(curve_points)
        if is_control:
            plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')

        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='tb-spline Curve')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('tB-spline Curve')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
