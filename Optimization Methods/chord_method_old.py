import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def chord_method_on_derivative(func, a, b, tol=1e-2, max_iter=10000):
    print(a, b)
    # Определяем переменную и функцию в виде sympy выражений
    x_sym = sp.symbols('x')
    func_sym = func(x_sym)

    # Находим производную от функции
    dfunc_sym = sp.diff(func_sym, x_sym)

    # Превращаем символьные выражения в функции для численных расчетов
    func_numeric = sp.lambdify(x_sym, func_sym, 'numpy')
    dfunc_numeric = sp.lambdify(x_sym, dfunc_sym, 'numpy')

    steps = []
    for _ in range(max_iter):
        dfa = dfunc_numeric(a)
        dfb = dfunc_numeric(b)

        # Если производные близки — меняем шаг, чтобы ускорить сходимость
        if abs(dfa - dfb) < tol:
            b = a + (b - a) / 2  # Уменьшение интервала

        # Вычисление новой точки по формуле метода хорд (по производной)
        x = a - (dfa * (a - b)) / (dfa - dfb)
        dfx = dfunc_numeric(x)

        if dfx > 0:
            b = x
        else:
            a = x

        steps.append((a, b, x, dfx))
        # Проверка на окончание поиска
        if abs(dfx) <= tol:
            return steps

    return steps

# Тестовая функция (кубическая)
def cubic(x):
    return 3 * x ** 2 - 15


# Начальные значения и вызов метода
a, b = -50, 50
steps = chord_method_on_derivative(cubic, a, b)
print(steps)
print(len(steps))

# Визуализация
fig, ax = plt.subplots()
x_vals = np.linspace(a - 0.01, b + 0.01, 1000)
y_vals = cubic(x_vals)

# Вычисление производной на интервале
x_sym = sp.symbols('x')
func_sym = cubic(x_sym)
dfunc_sym = sp.diff(func_sym, x_sym)
dfunc_numeric = sp.lambdify(x_sym, dfunc_sym, 'numpy')
y_prime_vals = dfunc_numeric(x_vals)

# Отображение функции (в качестве фона)
line_func, = ax.plot(x_vals, y_vals, label="Function", color='gray')

# Отображение производной
line_derivative, = ax.plot(x_vals, y_prime_vals, label="Derivative", color='orange')

# Линии для a, b, x (на графике производной)
line_a, = ax.plot([], [], 'r--', label="a")
line_b, = ax.plot([], [], 'b--', label="b")
point_x, = ax.plot([], [], 'go', label="x")

# Линия хорды (между a и b на графике производной)
chord_line, = ax.plot([], [], 'm-', label="Chord a-b")

# Настройки графика
ax.grid(True)
ax.legend()
ax.set_title("Chord Method on Derivative")

# Индекс текущего шага
step_idx = 0

# Функция обновления графика
def update_graph(step_idx):
    step = steps[step_idx]
    a, b, x, _ = step
    print(f"a={a}, b={b}, x={x}")

    # Обновление линий a, b и x на графике производной
    line_a.set_data([a, a], [0, dfunc_numeric(a)])
    line_b.set_data([b, b], [0, dfunc_numeric(b)])
    point_x.set_data([x], [dfunc_numeric(x)])

    # Линия хорды (одна хорда между a и b на графике производной)
    chord_line.set_data([a, b], [dfunc_numeric(a), dfunc_numeric(b)])

    # Масштабирование графика по текущим границам a и b
    ax.set_xlim(a - abs(a) * 0.1, b + abs(b) * 1.1)
    ax.set_ylim(min(dfunc_numeric(a), dfunc_numeric(b), dfunc_numeric(x)) - 1, max(dfunc_numeric(a), dfunc_numeric(b), dfunc_numeric(x)) + 1)

    ax.set_title(f"a={a:.5f}, b={b:.5f}, x={x:.5f}")

    fig.canvas.draw()

update_graph(step_idx)

# Функции для кнопок "Next" и "Previous"
def next_step(event):
    global step_idx
    if step_idx < len(steps) - 1:
        step_idx += 1
        update_graph(step_idx)

def prev_step(event):
    global step_idx
    if step_idx > 0:
        step_idx -= 1
        update_graph(step_idx)

# Создание кнопок
ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')

# Привязка кнопок к действиям
btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

plt.show()
