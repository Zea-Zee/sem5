import numpy as np
import math
# import matplotlib.pyplot as plt


def func(x):
    return x ** 2 + abs(math.sin(x))


def dichotomy_method(func, a, b, tol=1e-3, delta=1e-6):
    """
    Метод дихотомии для одномерной оптимизации.

    func: Целевая функция
    a, b: Интервал поиска минимума
    tol: Точность
    delta: Малый сдвиг для дихотомии
    """
    while (b - a) / 2.0 > tol:
        mid = (a + b) / 2.0
        x1 = mid - delta
        x2 = mid + delta

        if func(x1) < func(x2):
            b = x2
        else:
            a = x1

    return (a + b) / 2.0


def golden_section_method(func, a, b, tol=1e-3):
    """
    Метод золотого сечения для одномерной оптимизации.

    func: Целевая функция
    a, b: Интервал поиска минимума
    tol: Точность
    """
    phi = (1 + np.sqrt(5)) / 2  # Число золотого сечения

    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi

    while abs(b - a) > tol:
        if func(x1) < func(x2):
            b = x2
            x2 = x1
            x1 = b - (b - a) / phi
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) / phi

    return (a + b) / 2.0


def quadratic(x):
    return (x - 2)**2


#f(x)=(x−2)
#x=2
print(dichotomy_method(func, -10, 10))
print(golden_section_method(quadratic, -10, 10))
