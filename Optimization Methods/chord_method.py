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


def chord_method(func, a, b, eps=0.01):
    max_iters = 1000
    a_prev = a+1
    b_prev = b+1
    for i in range(max_iters):
        if i < 3:
            plt.plot([a, b], [func(a), func(b)], '--r', zorder=10)
            plt.scatter([a, b], [func(a), func(b)],
                        color='green', s=20, zorder=5)
            if (a_prev != a):
                plt.annotate(f'{i+1}-a', xy=(a, func(a)),
                             xytext=(a, func(a) + 0.4))
            if (b_prev != b):
                plt.annotate(f'{i+1}b', xy=(b, func(b)),
                             xytext=(b, func(b) + 0.4))
        f_a = func(a)
        f_b = func(b)
        x = b - (f_b * (b - a)) / (f_b - f_a)
        f_x = func(x)

        a_prev = a
        b_prev = b
        if f_x < 0:
            a = x
        else:
            b = x

        print(f"{i} - a: {a:.3f}, b: {b:.3f}, x:{x:.3f}, f_a: {
              f_a:.3f}, f_b: {f_b:.3f}, f_x: {f_x:.3f}, eps: {abs(b - a):.3f}")
        if (np.abs(func(x)) < eps):
            return x

        if (abs(b - a) < eps):
            return ((a + b) / 2)
    return ((a + b) / 2)


t = np.arange(-2, 1, 0.0001)
ans = chord_method(func5_chord, -2, 1, eps=0.001)
plt.axhline(0, color='black')
plt.plot(t, func5_chord(t), 'b', ans, func5_chord(ans), 'ro')
print(f"x: {ans:.3f}, y: {func5_chord(ans):.3f}")
plt.show()
