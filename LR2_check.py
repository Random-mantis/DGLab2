import math as m
import os

import matplotlib.pyplot as plt

import LR2

DATA_DIR = os.path.join(
    r"c:\Users\Kostya\Desktop\Important\Programs\VUS" +
    r"\Third Course\DigitalMethods",
    "data",
)
REPORT_PATH = os.path.join(DATA_DIR, "lr2_report.txt")
DEFAULT_EPS = 1e-6
_report_file = None


def data_path(name):
    return os.path.join(DATA_DIR, name)


def log(text=""):
    print(text, flush=True)
    if _report_file is not None:
        _report_file.write(text + "\n")
        _report_file.flush()


def variable_names(n):
    names = ["x", "y", "z", "u", "v", "w"]
    if n <= len(names):
        return names[:n]
    return ["x" + str(i) for i in range(n)]


def make_function(expression, names):
    code = "lambda " + ", ".join(names) + ": " + expression
    return eval(code, {"m": m, "math": m, "__builtins__": {}})


def read_problems(filename):
    lines = []
    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    index = 0
    problems = []
    while index < len(lines):
        n = int(lines[index])
        index += 1
        names = variable_names(n)
        f_exprs = lines[index:index + n]
        index += n
        funcs = [make_function(expr, names) for expr in f_exprs]
        problems.append(
            {
                "n": n,
                "names": names,
                "f_exprs": f_exprs,
                "funcs": funcs,
            }
        )
    return problems


def read_eps_list(filename):
    epsilons = []
    try:
        with open(filename, encoding="utf-8") as file:
            for line in file:
                line = line.strip().replace(",", ".")
                if not line or line.startswith("#"):
                    continue
                epsilons.append(float(line.split()[0]))
    except FileNotFoundError:
        pass
    return epsilons


def read_eps(epsilons, index=0):
    if index < len(epsilons):
        return epsilons[index]
    return DEFAULT_EPS


def plot_errors(errors_a, label_a, errors_b, label_b, title):
    fig, ax = plt.subplots()
    n_max = max(len(errors_a), len(errors_b), 1)
    if errors_a:
        ax.plot(range(1, len(errors_a) + 1), errors_a, marker="o",
                label=label_a, markersize=5)
    if errors_b:
        ax.plot(range(1, len(errors_b) + 1), errors_b, marker="o",
                label=label_b, markersize=5)
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Погрешность")
    if errors_a or errors_b:
        ax.set_yscale("log")
    if n_max <= 20:
        ax.set_xticks(list(range(1, n_max + 1)))
    else:
        step = max(1, n_max // 10)
        ax.set_xticks(list(range(1, n_max + 1, step)))
    ax.set_xlim(0.5, n_max + 0.5)
    ax.grid(True)
    if errors_a or errors_b:
        ax.legend()
    ax.set_title(title)
    plt.show(block=True)
    plt.close(fig)


def domain_for_expr(expr):
    left, right = 0.0, 3.0
    if "sqrt(1 - x" in expr:
        left, right = 0.0, 0.999
    if "tan" in expr:
        left, right = 0.0, 1.0
    return left, right


def report_method_result(name, funcs, root, errors, eps, scalar_func=None):
    if LR2.domain_error_happened():
        log(f"{name}: выход за область определения")
        LR2.clear_domain_error()
        return
    if scalar_func is not None:
        ok = LR2.has_real_root_scalar(scalar_func, root, eps)
    else:
        ok = LR2.has_real_solution(funcs, root, eps)
    if not ok:
        log(f"{name}: вещественного решения нет")
        return
    log(f"{name}: {root}, итер. {len(errors)}")
    if scalar_func is not None:
        log(f"g({name}) = {scalar_func(root)}")
    else:
        log(f"F({name}) = {LR2.evaluate_system(funcs, root)}")


def run_scalar(problem, eps, index, total):
    func = problem["funcs"][0]
    expr = problem["f_exprs"][0]
    left, right = domain_for_expr(expr)
    log("")
    log(f"[{index}/{total}] g(x) = 0: {expr}  |  eps = {eps}")
    x0 = LR2.g_x_graphic(func, left=left, right=right)
    log(f"x0 = {x0}")

    LR2.clear_domain_error()
    try:
        root_newton, err_newton = LR2.Newtone_method(
            func, eps, x0=x0, left=left, right=right,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        LR2.mark_domain_error()
        root_newton, err_newton = x0, []
    report_method_result("Ньютон", None, root_newton, err_newton, eps, func)

    LR2.clear_domain_error()
    try:
        root_iter, err_iter = LR2.iteration_method(
            func, eps, x0=x0, left=left, right=right,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        LR2.mark_domain_error()
        root_iter, err_iter = x0, []
    report_method_result("Итерация", None, root_iter, err_iter, eps, func)

    plot_errors(
        err_newton, "Ньютон", err_iter, "Простая итерация",
        f"Задача {index}",
    )


def run_system(problem, eps, index, total):
    n = problem["n"]
    funcs = problem["funcs"]
    log("")
    log(f"[{index}/{total}] система n = {n}  |  eps = {eps}")
    for expr in problem["f_exprs"]:
        log("  " + expr)
    if n > 2:
        x0 = None
    else:
        x0 = LR2.g_x_graphic2(funcs[0], funcs[1])
        log(f"x0 = {x0}")

    LR2.clear_domain_error()
    try:
        root_newton, err_newton = LR2.Newtone_system_method(
            funcs, eps, x0=x0, max_iterations=200,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        LR2.mark_domain_error()
        root_newton = x0 if x0 is not None else [0.0] * n
        err_newton = []
    report_method_result("Ньютон", funcs, root_newton, err_newton, eps)

    LR2.clear_domain_error()
    try:
        root_iter, err_iter = LR2.iteration_system_method(
            funcs, eps, x0=x0, max_iterations=500,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        LR2.mark_domain_error()
        root_iter = x0 if x0 is not None else [0.0] * n
        err_iter = []
    report_method_result("Итерация", funcs, root_iter, err_iter, eps)

    plot_errors(
        err_newton, "Ньютон", err_iter, "Простая итерация",
        f"Задача {index}",
    )


def main():
    global _report_file
    problems = read_problems(data_path("nonlinear.txt"))
    epsilons = read_eps_list(data_path("nonlinear_eps.txt"))
    log(f"Задач: {len(problems)}")
    _report_file = open(REPORT_PATH, "w", encoding="utf-8")
    try:
        for index, problem in enumerate(problems):
            eps = read_eps(epsilons, index)
            if problem["n"] == 1:
                run_scalar(problem, eps, index + 1, len(problems))
            else:
                run_system(problem, eps, index + 1, len(problems))
        log("Готово.")
    finally:
        _report_file.close()
        _report_file = None


if __name__ == "__main__":
    main()
