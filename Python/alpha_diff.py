import numpy as np
import matplotlib.pyplot as plt

from Python.b_spline import b_spline_curve
from Python.preset_figure import duck
from Python.t_2_b_spline_iterative import t_2_b_spline_curve


def plot_results(result1, result2, result3):
    plt.figure(figsize=(8, 6))

    # # Plot results of algorithm 1
    # plt.scatter(result1[:, 0], result1[:, 1], c='b', label='Algorithm 1')
    #
    # # Plot results of algorithm 2
    # plt.scatter(result2[:, 0], result2[:, 1], c='r', label='Algorithm 2')

    # Plot results of algorithm 3
    plt.scatter(result3[:, 0], result3[:, 1], c='g', label='Algorithm 3')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Results of Three Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()




alpha= np.pi/4
alpha= np.pi/2


control_points=duck()

control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
p = len(control_points)-1

def u():


    knots=[0,0,0]
    for i in range(3,p+1):
        knots+=[(i-2)*alpha]
    knots+=[(p-1)*alpha]*3
    # knots+=knots[0:2]

    return knots

def pi():


    knots=[]
    for i in range(p+3):
        knots+=[i*alpha]
    return knots

degree=2
knots=u()
num_points=1000
space_end=80

curve_points= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])


curve_points_1= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])
curve_points_1=curve_points[~np.all(curve_points_1==0, axis=1)]


curve_points_2= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])
curve_points_2=curve_points[~np.all(curve_points_2==0, axis=1)]


curve_points_3= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])
curve_points_3=curve_points[~np.all(curve_points_3==0, axis=1)]

control_points=duck()


plot_results(curve_points_1,curve_points_2,curve_points_3 )
control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
degree=2
num_points=1000

num_knots = len(control_points) + degree + 1
knots = np.linspace(0, 1, num_knots-4)
# knots = clamped_knot_vector(6,degree)
knots =np.concatenate([[0]*2,knots,[1]*2])
space_end=3
curve_points_b = np.array([b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, space_end, num_points)])



plot_results(curve_points_1,curve_points_2,curve_points_b )