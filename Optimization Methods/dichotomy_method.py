import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math


def quadratic(x):
    return (x - 2)**2


def qubiq(x):
    return x ** 3 + 13 * x ** 2 -4 * x + 17


def x2sinx(x):
    return x ** 2 + abs(np.sin(x))


func = x2sinx


def dichotomy_steps(func, a, b, tol=4 * 1e-2, delta=1e-2):
    steps = []
    while (b - a) / 2.0 > tol:
        mid = (a + b) / 2.0
        x1 = mid - delta
        x2 = mid + delta
        steps.append((a, b, x1, x2, func(x1), func(x2)))
        if func(x1) < func(x2):
            b = x2
        else:
            a = x1
    return steps


steps = dichotomy_steps(func, 0, 5)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

x_vals = np.linspace(0, 5, 400)
y_vals = func(x_vals)
line_func, = ax.plot(x_vals, y_vals, label="Function")

line_a, = ax.plot([], [], 'r--', label="a")
line_b, = ax.plot([], [], 'b--', label="b")
point_x1, = ax.plot([], [], 'go', label="x1")
point_x2, = ax.plot([], [], 'mo', label="x2")
delta_line1, = ax.plot([], [], 'g-', label="delta")
delta_line2, = ax.plot([], [], 'm-', label="delta")

ax.legend()
ax.set_title("Dichotomy Method Visualization")

step_idx = 0


def update_graph(step_idx):
    step = steps[step_idx]
    a, b, x1, x2, fx1, fx2 = step
    line_a.set_data([a, a], [0, func(a)])
    line_b.set_data([b, b], [0, func(b)])
    point_x1.set_data([x1], [func(x1)])
    point_x2.set_data([x2], [func(x2)])

    delta_line1.set_data([x1, x1], [0, fx1])
    delta_line2.set_data([x2, x2], [0, fx2])

    ax.set_xlim(a - 0.1, b + 0.1)
    ax.set_ylim(0, max(func(a), func(
        b), func(x1), func(x2)) + 0.5)


    if fx1 < fx2:
        comparison = "f(x1) < f(x2)"
    else:
        comparison = "f(x1) >= f(x2)"

    ax.set_title(f"a={a:.5f}, b={b:.5f}, x1={x1:.5f}, x2={x2:.5f} | {comparison}")

    fig.canvas.draw()


update_graph(step_idx)


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


ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')


btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

plt.show()
