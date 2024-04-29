from matplotlib import pyplot as plt

from Python.preset_figure import duck
from b_spline import  b_spline_curve
from t_b_spline_iterative import t_b_spline_curve
from t_2_b_spline_iterative import t_2_b_spline_curve
import numpy as np

control_points=duck()

control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
degree=2
num_points=1000

num_knots = len(control_points) + degree + 1
knots = np.linspace(0, 1, num_knots-4)
# knots = clamped_knot_vector(6,degree)
knots =np.concatenate([[0]*2,knots,[1]*2])
space_end=3
curve_points_b = np.array([b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, space_end, num_points)])


# curve_points_b = curve_points[~np.all(curve_points == 0, axis=1)]

curve_points_t_b= np.array([t_b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, space_end, num_points)])


# curve_points_t_b = curve_points[~np.all(curve_points == 0, axis=1)]


alpha= np.pi/4



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

knots=u()

curve_points= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])


curve_points_t_2_b =curve_points[~np.all(curve_points==0, axis=1)]


error1 = np.linalg.norm(curve_points_t_b - curve_points_b, axis=1)  # Error between result1 and result2
error2 = np.linalg.norm(curve_points_t_2_b - curve_points_b, axis=1)  # Error between result1 and result3
error3 = np.linalg.norm(curve_points_t_2_b - curve_points_t_b, axis=1)  # Error between result2 and result3

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(error1, label='Error between Algorithm 1 and 2')
plt.plot(error2, label='Error between Algorithm 1 and 3')
plt.plot(error3, label='Error between Algorithm 2 and 3')
plt.xlabel('Point Index')
plt.ylabel('Error')
plt.title('Error Comparison Between Algorithms')
plt.legend()
plt.grid(True)
plt.show()