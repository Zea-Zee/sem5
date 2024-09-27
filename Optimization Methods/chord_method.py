import sympy as sp
import numpy as np

def chord_method(func, a, b, tol=1e-6, max_iter=10000):
    # Определяем переменную и функцию в виде sympy выражений
    x_sym = sp.symbols('x')
    func_sym = func(x_sym)
    
    # Находим производную от функции
    dfunc_sym = sp.diff(func_sym, x_sym)
    
    # Превращаем символьные выражения в функции для численных расчетов
    dfunc_numeric = sp.lambdify(x_sym, dfunc_sym, 'numpy')
    
    steps = []
    for _ in range(max_iter):
        dfa = dfunc_numeric(a)
        dfb = dfunc_numeric(b)
        
        # Вычисление новой точки по формуле метода хорд
        x = a - (dfa * (a - b)) / (dfa - dfb)
        dfx = dfunc_numeric(x)
        
        steps.append((a, b, x, dfx))
        
        # Проверка на окончание поиска
        if abs(dfx) <= tol:
            return steps
        
        # Обновление границ в зависимости от знака производной
        if dfx > 0:
            b = x
        else:
            a = x
    
    return steps


# Пример функции
def cubic(x):
    return -x**3 + x**2 + x


# Поиск экстремума функции cubic
result = chord_method(cubic, -2, 0)
print(result[-1])  # Вывод последнего шага, где производная близка к нулю
