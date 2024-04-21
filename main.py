import math

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# Функція для обчислення коефіцієнтів c_{ki} для одного базисного вузла x_i
def compute_coefficients_for_node(x_i, y_i, phi_k, alpha=0.9):
    num_basis_functions = len(phi_k)

    A = np.zeros((num_basis_functions, num_basis_functions))
    b = np.zeros(num_basis_functions)

    # Заповнення матриці A та вектора b для обчислення коефіцієнтів
    for j in range(num_basis_functions):
        for k in range(num_basis_functions):
            A[j, k] = np.sum(phi_k[j](x_i - x_i) * phi_k[k](x_i - x_i))
        b[j] = np.sum(y_i * phi_k[j](x_i - x_i))

    # Додавання регуляризаційного члену до діагональних елементів матриці A
    A += alpha * np.eye(num_basis_functions)

    # Розв'язання системи лінійних рівнянь для знаходження коефіцієнтів
    coefficients = np.linalg.solve(A, b)
    return coefficients


# Функція для паралельного обчислення коефіцієнтів c_{ki} для всіх базисних вузлів
def compute_coefficients_parallel(x, y, phi_k):
    num_nodes = len(x)
    num_basis_functions = len(phi_k)

    # Використання Parallel та delayed для паралельного обчислення коефіцієнтів для кожного базисного вузла
    coefficients = Parallel(n_jobs=-1)(
        delayed(compute_coefficients_for_node)(x[i], y[i], phi_k) for i in range(num_nodes))
    return np.array(coefficients)


# Відображення функції f(x) та базисних функцій phi_k(x)
def plot_approximation(x, y, phi_k, coefficients):
    # Відрізок [a, b]
    a, b = np.min(x), np.max(x)

    # Візуалізація функції f(x)
    plt.scatter(x, y, label='f(x)')

    # Візуалізація апроксимації за допомогою базисних функцій
    x_values = np.linspace(a, b, 100)
    approximation = np.zeros_like(x_values)
    for k in range(len(phi_k)):
        approximation += coefficients[:, k] @ np.array([phi_k[k](x_i - x_values) for x_i in x])
        plt.plot(x_values, approximation, label=f'$\phi_{k}(x)$')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Апроксимація функції за допомогою базисних функцій')
    plt.grid(True)
    plt.show()


# Приклад використання
# Представимо, що ми вже маємо список координат x_i та відповідних значень функції y_i
# x = np.linspace(0, 5, 20)
# y = x**2
x_regular = [0, 0.951, -0.951, 0.588, -0.588]
y_regular = [1, 0.309, 0.309, -0.809, -0.809]

# Additional points near the center
x_center = [0, 0.4755, -0.4755, 0.294, -0.294]
y_center = [0.5, 0.1545, 0.1545, -0.4045, -0.4045]

# Combine x and y arrays into a single array of points
x=x_regular+x_center
y=y_regular+y_center

# Представимо, що phi_k - це список функцій базисних вузлів
# В цьому прикладі ми припустимо, що phi_k буде список з двох функцій
phi_k = [lambda x: x**2, lambda x: x**2]

# Обчислення коефіцієнтів c_{ki} за допомогою паралельних обчислень
coefficients = compute_coefficients_parallel(x, y, phi_k)

# Відображення результатів
plot_approximation(x, y, phi_k, coefficients)

import numpy as np

# Функція для обчислення коефіцієнтів c_{ki} для всіх базисних вузлів
def compute_coefficients(x, y, phi_k, alpha=math.pi/2):
    num_nodes = len(x)
    num_basis_functions = len(phi_k)

    coefficients = np.zeros((num_nodes, num_basis_functions))

    for i in range(num_nodes):
        A = np.zeros((num_basis_functions, num_basis_functions))
        b = np.zeros(num_basis_functions)

        for j in range(num_basis_functions):
            for k in range(num_basis_functions):
                A[j, k] = np.sum(phi_k[j](x[i] - x[i]) * phi_k[k](x[i] - x[i]))
            b[j] = np.sum(y[i] * phi_k[j](x[i] - x[i]))

        # Додавання регуляризаційного члену до діагональних елементів матриці A
        A += alpha * np.eye(num_basis_functions)

        coefficients[i] = np.linalg.solve(A, b)

    return coefficients


# Приклад використання
# Представимо, що ми вже маємо список координат x_i та відповідних значень функції y_i


# Представимо, що phi_k - це список функцій базисних вузлів
# В цьому прикладі ми припустимо, що phi_k буде список з двох функцій
phi_k = [lambda x: np.cos(x), lambda x: np.sin(x)]

# Обчислення коефіцієнтів c_{ki} без паралельних обчислень
coefficients_without_parallel = compute_coefficients(x, y, phi_k)

print("Коефіцієнти c_{ki} без використання паралельних обчислень:")
print(coefficients_without_parallel)


import time

# Обчислення коефіцієнтів з використанням паралельних обчислень
start_time_parallel = time.time()
coefficients_parallel = compute_coefficients_parallel(x, y, phi_k)
end_time_parallel = time.time()

# Обчислення коефіцієнтів без використання паралельних обчислень
start_time_without_parallel = time.time()
coefficients_without_parallel = compute_coefficients(x, y, phi_k)
end_time_without_parallel = time.time()

# Порівняння часу виконання
print("Час виконання з використанням паралельних обчислень:", end_time_parallel - start_time_parallel)
print("Час виконання без використання паралельних обчислень:", end_time_without_parallel - start_time_without_parallel)
