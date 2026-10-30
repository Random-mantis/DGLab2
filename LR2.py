import math as m

import matplotlib.pyplot as plt
import numpy as np

import Gauss

_domain_error = False


def clear_domain_error():
    global _domain_error
    _domain_error = False


def domain_error_happened():
    return _domain_error


def mark_domain_error():
    global _domain_error
    _domain_error = True


def g_x_graphic(func, left=0.0, right=3.0):
    clicked = []

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        clicked.append(float(event.xdata))
        plt.close(event.canvas.figure)

    dots = np.linspace(left, right, 800)
    values = []
    for x in dots:
        try:
            values.append(float(func(x)))
        except (ValueError, ZeroDivisionError, OverflowError):
            values.append(np.nan)
    fig, ax = plt.subplots()
    ax.plot(dots, values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(True)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    if not clicked:
        raise RuntimeError("Точка на графике не выбрана")
    return clicked[0]


def g_x_graphic2(func1, func2, left=-4.0, right=4.0):
    clicked = []

    def on_click(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        clicked.append((float(event.xdata), float(event.ydata)))
        plt.close(event.canvas.figure)

    x = np.linspace(left, right, 400)
    y = np.linspace(left, right, 400)
    X, Y = np.meshgrid(x, y)
    F1 = np.zeros_like(X)
    F2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                F1[i, j] = func1(X[i, j], Y[i, j])
                F2[i, j] = func2(X[i, j], Y[i, j])
            except (ValueError, ZeroDivisionError, OverflowError):
                F1[i, j] = np.nan
                F2[i, j] = np.nan
    fig, ax = plt.subplots()
    try:
        if np.nanmin(F1) * np.nanmax(F1) <= 0:
            ax.contour(X, Y, F1, levels=[0.0], colors="blue")
    except (ValueError, TypeError):
        pass
    try:
        if np.nanmin(F2) * np.nanmax(F2) <= 0:
            ax.contour(X, Y, F2, levels=[0.0], colors="red")
    except (ValueError, TypeError):
        pass
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlim(left, right)
    ax.set_ylim(left, right)
    ax.grid(True)
    ax.set_aspect("equal")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    if not clicked:
        raise RuntimeError("Точка на графике не выбрана")
    return np.array(clicked[0], dtype=float)


def g_prime(func, x, h=1e-8):
    try:
        return (func(x + h) - func(x - h)) / (2 * h)
    except (ValueError, ZeroDivisionError, OverflowError):
        mark_domain_error()
        return float("nan")


def evaluate_system(funcs, values):
    result = np.empty(len(funcs), dtype=float)
    for i in range(len(funcs)):
        try:
            result[i] = funcs[i](*values)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            mark_domain_error()
            result[i] = float("nan")
    return result


def residual_of_system(funcs, values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return m.inf
    return vector_norm(evaluate_system(funcs, values))


def has_real_solution(funcs, values, epsilon):
    residual = residual_of_system(funcs, values)
    limit = max(1e-3, 100.0 * float(epsilon))
    return m.isfinite(residual) and residual <= limit


def has_real_root_scalar(func, value, epsilon):
    try:
        residual = abs(float(func(value)))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        mark_domain_error()
        return False
    if not m.isfinite(residual):
        return False
    limit = max(1e-3, 100.0 * float(epsilon))
    return residual <= limit


def jacobian_matrix(funcs, x, h=1e-8):
    n = len(funcs)
    matrix = np.zeros((n, n), dtype=float)
    for j in range(n):
        x_right = np.array(x, dtype=float).copy()
        x_left = np.array(x, dtype=float).copy()
        x_right[j] += h
        x_left[j] -= h
        f_right = evaluate_system(funcs, x_right)
        f_left = evaluate_system(funcs, x_left)
        matrix[:, j] = (f_right - f_left) / (2 * h)
    return matrix


def vector_norm(values):
    result = 0.0
    for value in values:
        value = float(value)
        if not m.isfinite(value):
            return m.inf
        result += value * value
    return m.sqrt(result)


def Norm(lhs, rhs, epsilon):
    for i in range(len(lhs)):
        if abs(lhs[i] - rhs[i]) > epsilon or lhs[i] < epsilon or rhs[i] < epsilon:
            return False
    return True


def Yakobi_Lambda(funcs, x, h=1e-8):
    n = len(funcs)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        x_right = np.array(x, dtype=float).copy()
        x_left = np.array(x, dtype=float).copy()
        x_right[i] += h
        x_left[i] -= h
        try:
            left = funcs[i](*x_right)
            right = funcs[i](*x_left)
            deriv = (left - right) / (2 * h)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            mark_domain_error()
            deriv = float("nan")
        if not m.isfinite(deriv) or abs(deriv) < 1e-14:
            deriv = 1e-14 if (m.isfinite(deriv) and deriv >= 0) else -1e-14
        matrix[i][i] = 1.0 / deriv
    return matrix


def iteration_lambda(func, x0, left=-2.0, right=3.0, samples=1000):
    span = max(0.5, 0.25 * (right - left))
    a = max(left, float(x0) - span)
    b = min(right, float(x0) + span)
    if b <= a:
        a, b = left, right
    points = np.linspace(a, b, samples)
    max_deriv = 0.0
    for point in points:
        deriv = g_prime(func, point)
        if m.isfinite(deriv):
            max_deriv = max(max_deriv, abs(deriv))
    if max_deriv < 1e-14:
        raise ZeroDivisionError()
    deriv0 = g_prime(func, x0)
    if not m.isfinite(deriv0) or abs(deriv0) < 1e-14:
        deriv0 = max_deriv
    return m.copysign(1.0, deriv0) / max_deriv


def iteration_method(func, epsilon, x0=None, max_iterations=100000,
                     left=-2.0, right=3.0):
    if x0 is None:
        x0 = g_x_graphic(func, left=left, right=right)
    lam = iteration_lambda(func, x0, left=left, right=right)
    x_last = float(x0)
    errors = []
    for _ in range(max_iterations):
        try:
            value = func(x_last)
        except (ValueError, ZeroDivisionError, OverflowError):
            mark_domain_error()
            return x_last, errors
        if not m.isfinite(value):
            return x_last, errors
        x_next = x_last - lam * value
        if not m.isfinite(x_next):
            return x_last, errors
        errors.append(abs(x_next - x_last))
        if abs(x_next - x_last) <= epsilon:
            return x_next, errors
        x_last = x_next
    return x_last, errors


def Newtone_method(func, epsilon, x0=None, max_iterations=1000,
                   left=-2.0, right=3.0):
    if x0 is None:
        x0 = g_x_graphic(func, left=left, right=right)
    x_last = float(x0)
    errors = []
    for _ in range(max_iterations):
        try:
            value = func(x_last)
        except (ValueError, ZeroDivisionError, OverflowError):
            mark_domain_error()
            return x_last, errors
        deriv = g_prime(func, x_last)
        if not m.isfinite(deriv) or abs(deriv) < 1e-14:
            raise ZeroDivisionError()
        x_next = x_last - value / deriv
        if not m.isfinite(x_next):
            return x_last, errors
        errors.append(abs(x_next - x_last))
        if abs(x_next - x_last) <= epsilon:
            return x_next, errors
        x_last = x_next
    return x_last, errors


def iteration_direction(funcs, x, target):
    Lambda = Yakobi_Lambda(funcs, x)
    return -(Lambda @ (evaluate_system(funcs, x) - target))


def newton_direction(funcs, x, target, epsilon=1e-8):
    jac = np.array(jacobian_matrix(funcs, x), dtype=float)
    right = -np.array(evaluate_system(funcs, x) - target, dtype=float)
    matrix = jac.T @ jac
    vector = jac.T @ right
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        return np.full(len(right), float("nan"), dtype=float)
    try:
        delta = np.linalg.solve(matrix, vector)
        return np.array(delta, dtype=float)
    except np.linalg.LinAlgError:
        pass
    try:
        delta, _ = Gauss.Zeidel(matrix, vector, epsilon)
        return np.array([float(value) for value in delta], dtype=float)
    except (ZeroDivisionError, ValueError, OverflowError):
        return np.full(len(right), float("nan"), dtype=float)


def solve_with_steps(funcs, x0, target, epsilon, direction_fn, max_iterations):
    x_last = np.array(x0, dtype=float).copy()
    errors = []
    for _ in range(max_iterations):
        current = vector_norm(evaluate_system(funcs, x_last) - target)
        if not m.isfinite(current):
            break
        if current < 1e-14:
            result = np.array(x_last, dtype=float).copy()
            for i in range(len(result)):
                if abs(result[i]) < 1e-12:
                    result[i] = 0.0
            errors.append(0.0)
            return result, errors
        direction = direction_fn(funcs, x_last, target)
        if not np.all(np.isfinite(direction)):
            break
        alpha = 1.0
        x_next = None
        residual_next = m.inf
        growth_limit = max(2.0 * current, current + 1.0)
        while alpha >= 1e-8:
            candidate = x_last + alpha * direction
            trial = vector_norm(evaluate_system(funcs, candidate) - target)
            if m.isfinite(trial) and trial <= growth_limit:
                x_next = candidate
                residual_next = trial
                break
            alpha *= 0.5
        if x_next is None:
            break
        step = vector_norm(x_next - x_last)
        errors.append(step)
        if residual_next <= epsilon:
            return x_next, errors
        if step <= epsilon:
            if residual_next <= max(1e-3, 100.0 * epsilon):
                return x_next, errors
            break
        x_last = x_next
    return x_last, errors


def continuation_method(funcs, epsilon, direction_fn, stages=50,
                        max_iterations=1000):
    n = len(funcs)
    current = 0.1 * np.arange(1, n + 1, dtype=float)
    f_start = evaluate_system(funcs, current)
    all_errors = []
    for k in range(1, stages + 1):
        t = k / stages
        target = (1.0 - t) * f_start
        current, errors = solve_with_steps(
            funcs,
            current,
            target,
            epsilon,
            direction_fn,
            max_iterations,
        )
        all_errors.extend(errors)
    return current, all_errors


def iteration_system_method(funcs, epsilon, x0=None, max_iterations=1000):
    n = len(funcs)
    if n > 2:
        return continuation_method(
            funcs,
            epsilon,
            iteration_direction,
            stages=50,
            max_iterations=min(max_iterations, 2000),
        )
    if x0 is None:
        if n == 2:
            x0 = g_x_graphic2(funcs[0], funcs[1])
        else:
            x0 = [g_x_graphic(funcs[0])]
    return solve_with_steps(
        funcs,
        x0,
        np.zeros(n, dtype=float),
        epsilon,
        iteration_direction,
        max_iterations,
    )


def Newtone_system_method(funcs, epsilon, x0=None, max_iterations=1000):
    n = len(funcs)
    if n > 2:
        return continuation_method(
            funcs,
            epsilon,
            newton_direction,
            stages=50,
            max_iterations=min(max_iterations, 2000),
        )
    if x0 is None:
        if n == 2:
            x0 = g_x_graphic2(funcs[0], funcs[1])
        else:
            x0 = [g_x_graphic(funcs[0])]
    return solve_with_steps(
        funcs,
        x0,
        np.zeros(n, dtype=float),
        epsilon,
        newton_direction,
        max_iterations,
    )
