# Реалізація методу найменших квадратів для двох змінних (варіант 15)

# Вхідні дані (варіант 15 — з урахуванням таблиці)
x1 = [0, 0, 0, 1, 2, 2, 2]
x2 = [1.5, 2.5, 3.5, 1.5, 1.5, 2.5, 2.5]
n = 15
y = [2.3, 4 + 0.3 * n, 2 - 0.1 * n, 5 - 0.2 * n, 6.1 + 0.2 * n, 6.5 - 0.1 * n, 7.2]

# Обчислення необхідних сум
sum_x1 = sum(x1)
sum_x2 = sum(x2)
sum_y = sum(y)
sum_x1y = sum(x1[i]*y[i] for i in range(len(y)))
sum_x2y = sum(x2[i]*y[i] for i in range(len(y)))
sum_x1x1 = sum(x1[i]**2 for i in range(len(y)))
sum_x2x2 = sum(x2[i]**2 for i in range(len(y)))
sum_x1x2 = sum(x1[i]*x2[i] for i in range(len(y)))

# Формування та розв'язання системи лінійних рівнянь
import numpy as np

A = [
    [len(y), sum_x1, sum_x2],
    [sum_x1, sum_x1x1, sum_x1x2],
    [sum_x2, sum_x1x2, sum_x2x2]
]

B = [sum_y, sum_x1y, sum_x2y]

a0, a1, a2 = np.linalg.solve(A, B)

# Виведення знайденої залежності
print(f'Залежність має вигляд: y = {a0:.2f} + {a1:.2f}*x1 + {a2:.2f}*x2')

# Знаходження значення у точці x1=1.5, x2=3
y_test = a0 + a1*1.5 + a2*3
print(f'Значення функції в точці x1=1.5, x2=3: {y_test:.2f}')

# Обчислення коефіцієнта детермінації R2
y_mean = sum_y / len(y)
SS_tot = sum((yi - y_mean)**2 for yi in y)
SS_res = sum((y[i] - (a0 + a1*x1[i] + a2*x2[i]))**2 for i in range(len(y)))
R2 = 1 - SS_res / SS_tot
print(f'Коефіцієнт детермінації R2: {R2:.4f}')


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, color='red', label='Експериментальні точки')


x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1), max(x1), 20), np.linspace(min(x2), max(x2), 20))
y_grid = a0 + a1 * x1_grid + a2 * x2_grid
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5)


x_special = 1.5
x2_special = 3
y_special = a0 + a1 * x_special + a2 * x2_special
ax.scatter(x_special, x2_special, y_special, color='blue', s=100, label='Точка (1.5, 3)')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Графік залежності між параметрами')

ax.legend()
plt.show()