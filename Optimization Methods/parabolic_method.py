import numpy as np
import matplotlib.pyplot as plt


def func1(x):
    return -np.sqrt((np.cos(x)**2)) + x


def func2(x):
    return -((x - 2) ** 2 + 1)


def func3(x):
    return x**3


def func4(x):
    return np.sin(x)**2


def func5(x):
    return x**2 + np.sin(2*x)


def func5_chord(x):
    return 2*x + 2*np.cos(2*x)


def parabolic_method(func, x1, x2, x3, eps=0.01):
    iters = []
    for i in range(1000):
        f_x1 = func(x1)
        f_x2 = func(x2)
        f_x3 = func(x3)

        x_sh = (1/2)*(x1 + x2 - ((f_x2 - f_x1) * (x3 - x2))/(x2 - x1) /
                      ((f_x3 - f_x1)/(x3 - x1) - (f_x2 - f_x1)/(x2 - x1)))

        if i != 0:
            if np.abs(x_sh - x2) < eps:
                print(f'Iters - {i}')
                return x_sh

        f_x_sh = func(x_sh)

        if i < 5:
            coefs = np.polyfit([x1, x2, x3], [f_x1, f_x2, f_x3], 2)
            p = np.poly1d(coefs)
            x_parabola = np.linspace(
                min(x1, x3) - 0.5, max(x1, x3) + 0.5, 1000)
            plt.plot(x_parabola, p(x_parabola), '--r')
            plt.scatter([x1, x2, x3], [func(x1), func(x2),
                        func(x3)], color='green', s=20, zorder=5)
            plt.annotate(f'{i+1}-x1', xy=(x1, func(x1)),
                         xytext=(x1, func(x1) + 0.3 * i))
            plt.annotate(f'{i+1}-x2', xy=(x2, func(x2)),
                         xytext=(x2, func(x2) + 0.3 * i))
            plt.annotate(f'{i+1}-x3', xy=(x3, func(x3)),
                         xytext=(x3, func(x3) + 0.3 * i))
            # plt.plot([x1, x2, x3], [func(x1), func(x2), func(x3)], 'go')

        if f_x_sh < f_x2:
            if x_sh > x2:
                x1 = x2
                x2 = x_sh
            else:
                x3 = x2
                x2 = x_sh
        else:
            if x_sh > x2:
                x3 = x_sh
            else:
                x1 = x_sh
    print(f'Iters - {i}')
    return x_sh


t = np.arange(-2, 1, 0.0001)
ans = parabolic_method(func5, -2, -1.5, 1, eps=0.001)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.plot(t, func5(t), 'b', ans, func5(ans), 'ro')
print(f"x: {ans:.3f}, y: {func5(ans):.3f}")
plt.show()
