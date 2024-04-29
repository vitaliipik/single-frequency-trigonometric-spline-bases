import re

import numpy as np
from scipy.interpolate import CubicSpline
from plotting import plot_basis
from preset_figure import spiral, duck, star


def basis_function(i, k, t, knots,alpha):
    """
    Calculate the i-th B-spline basis function of order k at parameter t
    """
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

    # denom1 = knots[i + k - 1] - knots[i]
    # denom2 = knots[i + k] - knots[i + 1]
    # alpha=np.pi*3/2

    # alpha=np.pi/2
    denom2= lambda i:2*np.sin(t-knots[i+1])*(np.cos(alpha)-1)
    coeff1 =lambda i: 0.0 if alpha == 0.0 else (np.sin((t - knots[i])/2)*np.cos((alpha-t + knots[i])/2)) / np.sin(alpha/2)
    top=lambda i: np.sin(t-knots[i+1]+alpha)-np.sin(t-knots[i+1])-np.sin(alpha)

    coeff2 =lambda i: 0.0 if denom2(i) == 0.0 else top(i) /denom2(i)

    if k==2:
        coeff1=coeff2

    p=1-coeff1(i+1)

    return coeff1(i) * basis_function(i, k - 1, t, knots,alpha) + p * basis_function(i + 1, k - 1, t, knots,alpha)


def t_2_b_spline_curve(control_points, degree, knots, t,alpha):

    """
    Calculate the B-spline curve at parameter t
    """
    n = len(control_points)+2-degree # number of control points
    result = np.zeros_like(control_points[0], dtype=float)  # Initialize result with float dtype

    for i in range(n):
        basis = basis_function(i, degree, t, knots,alpha)
        result += control_points[i] * basis

    return result


# Example usage:
control_points = np.array([[0, 0], [1, 3], [2, 4], [3, 4], [4, 0],[6, 2]])
degree = 5

knots = np.array([1, 1, 1, 0, 1, 2, 3, 3, 4, 4,4,4])  # Example knot vector



# Example usage:
control_points = np.array([[0, 0], [0,1], [3, 4], [6, 0], [7, 4]])
control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)

degree =2
knots = np.array([0,0,0, 1, 2, 3, 4, 3, 3,3])  # Example knot vector

# knots = np.array([0, 0,  0,   1,2, 3,  3, 3 ])
knots = np.array([0, 0,  0,   1,2, 3,  4,4 ,4])
knots = np.array([0, 0,  0,0,1,   2,3, 4, 5,5 ,5,5])
# knots = np.array([0, 1,  2,   3,4, 5, 6,7,8,9,10])
num_knots = len(control_points) + degree + 1
knots = np.linspace(0, 1, num_knots-4)
# knots = clamped_knot_vector(6,degree)
knots =np.concatenate([[0]*2,knots,[1]*2])


# Example usage
# alpha = np.pi/2
# alpha= np.pi*3/4
alpha= np.pi/4

# knots = N1_squared(alpha, p)
# knots = N2_squared(alpha, p)

# print("N0^2:", N0_sq)
# print("N1^2:", N1_sq)
# print("N2^2:", N2_sq)
control_points=star()
# control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
# knots=u = np.linspace(0, alpha, p + 3)
# knots=np.array([0,0,0,1*alpha,2*alpha,3*alpha,4*alpha,4*alpha,4*alpha,4*alpha])
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
# knots=pi()



# knots=knots[:-1]
# knots=np.array([0,0,0,1*alpha,2*alpha,3*alpha,4*alpha,5*alpha,5*alpha,5*alpha])
# knots=np.array([0,alpha,2*alpha,3*alpha,4*alpha,5*alpha,6*alpha,7*alpha])


#

# Generate points along the curve
import time

start = time.time()




num_points = 3000
curve_points= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, 160, num_points)])


end = time.time()
print(end - start)
curve_points =curve_points[~np.all(curve_points==0, axis=1)]
# curve_points=curve_points[:333]

plot_basis(control_points,curve_points)