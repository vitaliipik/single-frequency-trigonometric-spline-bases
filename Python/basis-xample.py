import numpy as np
import matplotlib.pyplot as plt

def interpolate_control_points(control_points):
    x_values = [point[0] for point in control_points]
    y_values = [point[1] for point in control_points]
    coeffs = np.polyfit(x_values, y_values, 2)  # Quadratic polynomial interpolation
    return coeffs

def TB_like_curve(control_points, t):
    # Interpolate control points to get the coefficients of the parabolic segment
    coeffs = interpolate_control_points(control_points)
    # Calculate x and y coordinates using the quadratic equation
    x = (control_points[0][0] - control_points[1][0]) * np.cos(t) + control_points[1][0]
    y = coeffs[2] * ((control_points[0][0] - control_points[1][0]) * np.cos(t) + control_points[1][0])**2 + \
        coeffs[1] * ((control_points[0][0] - control_points[1][0]) * np.cos(t) + control_points[1][0]) + coeffs[0]
    return x, y

# Example control points
control_points = [(1, 2), (2, 4), (3, 6)]

# Generate TB-like curve points
t_values = np.linspace(0, np.pi/2, 100)
x_curve, y_curve = TB_like_curve(control_points, t_values)

# Plot TB-like curve
plt.figure(figsize=(8, 6))
plt.plot(x_curve, y_curve, label='TB-like Curve', color='blue')

# Plot control points
x_control = [point[0] for point in control_points]
y_control = [point[1] for point in control_points]
plt.scatter(x_control, y_control, color='red', label='Control Points')

plt.xlabel('x')
plt.ylabel('y')
plt.title('TB-like Curve passing through Control Points')
plt.legend()
plt.grid(True)
plt.show()
