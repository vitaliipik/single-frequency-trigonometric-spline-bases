from matplotlib import pyplot as plt

from Python.plotting import plot_basis
from Python.preset_figure import duck, gerb, butterfly, star
from b_spline import  b_spline_curve
from t_b_spline_iterative import t_b_spline_curve
from t_2_b_spline_iterative import t_2_b_spline_curve
import numpy as np
from scipy.interpolate import BSpline, splrep, splev, splprep


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

control_points=duck()
control_points=butterfly()
control_points=star()
control_points= np.array([[0, 0], [0,1], [3, 4], [6, 0], [7, 4]])

# control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
degree=2
num_points=1000
alpha= np.pi/4
# alpha= 1e-10
alpha= 0.63

p = len(control_points)-1




knots=u()
space_end=5
# t=control_points[:,0].reshape(-1,1)
x=control_points[:,0]
y=control_points[:,1]
# x = np. linspace(0, 10, 10)
# y = np. sin(x)
sorted_indices = np.argsort(control_points[:,0])
x_sorted = control_points[:,0][sorted_indices]
y_sorted = control_points[:,1][sorted_indices]
xmin, xmax = x_sorted.min(), x_sorted.max()
t = range(len(control_points))
t = knots
space= np.linspace(xmin, xmax, num_points)

degree = 2
x_regular = [0, 0.951, -0.951, 0.588, -0.588]
y_regular = [1, 0.309, 0.309, -0.809, -0.809]

# Additional points near the center
x_center = [0, 0.4755, -0.4755, 0.294, -0.294]
y_center = [0.5, 0.1545, 0.1545, -0.4045, -0.4045]

# Combine x and y arrays into a single array of points
# x = x_regular + x_center
# y = y_regular + y_center
control_points = [[x_i, y_i] for x_i, y_i in zip(x, y)]
# points = control_points + control_points[0:degree + 1]
points = np.array(control_points)
n_points = len(points)
# x = points[:,0]
# y = points[:,1]

t = range(len(x))
t=knots[1:-2]
ipl_t = np.linspace(0, space_end, 1000)

# x_tup = splrep(t, x, k=degree)
# y_tup = splrep(t, y, k=degree)
# x_list = list(x_tup)
# xl = x.tolist()
# x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
#
# y_list = list(y_tup)
# yl = y.tolist()
# y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

# x_i = splev(ipl_t, x_list)
# y_i = splev(ipl_t, y_list)

# x = np.array([0, 0.951, -0.951, 0.588, -0.588,0, 0.4755, -0.4755, 0.294, -0.294])
# y = np.array([1, 0.309, 0.309, -0.809, -0.809,0.5, 0.1545, 0.1545, -0.4045, -0.4045])

sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# Define the degree of the B-spline
degree = 2

# Define the number of control points
num_control_points = len(x) - degree - 1

# Create a B-spline object
t, c = splprep([x, y],k=degree,s=0)

# Evaluate the B-spline at some points
x_i = np.linspace(0, 1, 1000,endpoint=True)
# y_i = BSpline(t, c, degree)(x_i)

# curve_points = np.array([[t,z] for t,z in zip(x_i,y_i)])
curve_points =splev(x_i,t)
curve_points=np.array(curve_points).T
control_points=np.array(control_points)
# curve_points_b = curve_points[~np.all(curve_points == 0, axis=1)]
curve_points_b = curve_points
plot_basis(np.array(control_points),curve_points_b)




# curve_points_t_b= np.array([t_b_spline_curve(control_points, degree, knots, t) for t in np.linspace(0, space_end, num_points)])


# curve_points_t_b = curve_points[~np.all(curve_points == 0, axis=1)]



degree=2

control_points=duck()
control_points = np.concatenate((control_points ,control_points [0:2]), axis=0)
def u():


    knots=[0,0,0]
    for i in range(3,p+1):
        knots+=[(i-2)*alpha]
    knots+=[(p-1)*alpha]*3
    # knots+=knots[0:2]

    return knots
p = len(control_points)-2
num_points=310
space_end=50
alpha=1

knots=u()
# knots=t
# alpha=np.pi/4

curve_points_b= np.array([ b_spline_curve(control_points[:-1], degree, knots, t) for t in np.linspace(0, space_end, num_points)])
curve_points_b =curve_points_b[~np.all(curve_points_b==0, axis=1)]



plot_basis(curve_points_b,control_points)
p = len(control_points)-1
num_points=1010
space_end=400
alpha=1

knots=u()
curve_points= np.array([t_2_b_spline_curve(control_points, degree, knots, t, alpha) for t in np.linspace(0, space_end, num_points)])

curve_points_t_2_b =curve_points[~np.all(curve_points==0, axis=1)]

curve_points_t_2_b=curve_points_t_2_b[:-6]


# curve_points_t_2_b =curve_points
plot_basis(control_points,curve_points_b,is_control=False)
plot_basis(control_points,curve_points_t_2_b,is_control=False)
plot_basis(curve_points_b,control_points)
plot_basis(curve_points_t_2_b,control_points)

# error1 = np.linalg.norm(curve_points_t_b - curve_points_b, axis=1)  # Error between result1 and result2
error2 = np.linalg.norm(curve_points_t_2_b - curve_points_b, axis=1)  # Error between result1 and result3
error2 = error2[5:]
# error3 = np.linalg.norm(curve_points_t_2_b - curve_points_t_b, axis=1)  # Error between result2 and result3
print(np.sum(error2)/len(error2))
# Plotting
plt.figure(figsize=(8, 6))
# plt.plot(error1, label='Error between Algorithm 1 and 2')
plt.plot(error2, label='Error between scipy and t-2-b-spline')
# plt.plot(error3, label='Error between Algorithm 2 and 3')
plt.xlabel('Point Index')
plt.ylabel('Error')
plt.title('Error Comparison')
plt.legend()
plt.grid(True)
plt.show()