import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class TrigSpline:
    def __init__(self, data_x, data_y, num_basis, omega):
        self.data_x = data_x
        self.data_y = data_y
        self.num_basis = num_basis
        self.omega = omega
        self.K = 2 * num_basis  # Total number of basis functions (cosine and sine)

    def basis_functions(self, x):
        basis = np.empty((len(x), self.K))
        for i in range(1, self.num_basis + 1):
            basis[:, 2 * i - 2] = np.cos(i * self.omega * x)
            basis[:, 2 * i - 1] = np.sin(i * self.omega * x)
        return basis

    def spline_function(self, coeffs, x):
        basis = self.basis_functions(x)
        return np.dot(basis, coeffs)

    def error_function(self, coeffs):
        y_pred = self.spline_function(coeffs, self.data_x)
        return np.sum((y_pred - self.data_y) ** 2)

    def shape_preservation_constraint(self, coeffs):
        # Example of a shape preservation constraint (monotonicity)
        # This constraint ensures that the spline is monotonically increasing
        spline_values = self.spline_function(coeffs, self.data_x)
        return np.diff(spline_values).min()

    def optimize(self):
        initial_guess = np.zeros(self.K)
        constraints = ({'type': 'ineq', 'fun': self.shape_preservation_constraint})
        result = minimize(self.error_function, initial_guess, constraints=constraints)
        return result.x

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data_x, self.data_y, color='red', label='Data')
        x_vals = np.linspace(min(self.data_x), max(self.data_x), 10)
        spline_coeffs = self.optimize()
        spline_vals = self.spline_function(spline_coeffs, x_vals)
        plt.plot(x_vals, spline_vals, color='blue', label='Spline')

        # Plot shape preservation constraint (for demonstration)
        plt.plot(x_vals[:-1], np.diff(spline_vals), color='green', label='Shape Preservation Constraint')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trig Spline with Shape Preservation')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
data_x = np.array([0, 1, 2, 3, 4])
data_y = np.array([0, 1, 0.5, -1, 0])
num_basis = 5
omega = 20 * np.pi  # Frequency parameter

trig_spline = TrigSpline(data_x, data_y, num_basis, omega)
trig_spline.plot()
