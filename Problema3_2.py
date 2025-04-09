import numpy as np
import pyomo.environ as pyo
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

# ==== SIMPLEX ESTÁNDAR ====
def simplex_standard(A, b, c):
    m, n = A.shape
    A_aug = np.hstack((A, np.eye(m)))
    c_aug = np.concatenate((-c, np.zeros(m)))
    tableau = np.zeros((m+1, n+m+1))
    tableau[1:, :-1] = A_aug
    tableau[1:, -1] = b
    tableau[0, :-1] = c_aug
    basic_vars = list(range(n, n+m))

    iterations = 0
    while np.any(tableau[0, :-1] < 0):
        iterations += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [
            tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(1, tableau.shape[0])
        ]
        if all(r == np.inf for r in ratios): break
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col

    Z = -tableau[0, -1]
    return Z, iterations

# ==== SIMPLEX DOS FASES ====
def simplex_two_phase(A, b, c):
    m, n = A.shape
    A1 = np.hstack((A, np.eye(m)))  # Artificiales
    c1 = np.concatenate((np.zeros(n), np.ones(m)))
    tableau = np.zeros((m+1, n+m+1))
    tableau[1:, :-1] = A1
    tableau[1:, -1] = b
    tableau[0, :-1] = -c1
    tableau[0, :] = -np.sum(tableau[1:, :], axis=0)

    basic_vars = list(range(n, n + m))
    iter1 = 0
    while np.any(tableau[0, :-1] < 0):
        iter1 += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [
            tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(1, tableau.shape[0])
        ]
        if all(r == np.inf for r in ratios): break
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col

    if abs(tableau[0, -1]) > 1e-5:
        return None, iter1 + 0  # no factible

    # Fase II
    tableau = tableau[:, :n+m+1]
    tableau[0, :] = 0
    tableau[0, :n] = -c
    for i, var in enumerate(basic_vars):
        if var < n:
            tableau[0] += tableau[0, var] * tableau[i+1]
    iter2 = 0
    while np.any(tableau[0, :-1] < 0):
        iter2 += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [
            tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(1, tableau.shape[0])
        ]
        if all(r == np.inf for r in ratios): break
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col

    Z = tableau[0, -1]
    return Z, iter1 + iter2

# ==== GLPK + PYOMO ====
def solve_glpk(A, b, c):
    model = pyo.ConcreteModel()
    model.n = pyo.RangeSet(0, len(c)-1)
    model.x = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=sum(c[i] * model.x[i] for i in model.n), sense=pyo.maximize)
    model.constraints = pyo.ConstraintList()
    for j in range(len(b)):
        model.constraints.add(expr=sum(A[j][i] * model.x[i] for i in range(len(c))) <= b[j])
    solver = pyo.SolverFactory('glpk')
    result = solver.solve(model, tee=False)
    return pyo.value(model.obj)

# ==== PRUEBA DE RENDIMIENTO ====
def benchmark(sizes, n_runs=3):
    results = []

    for size in sizes:
        print(f"\nProbando tamaño {size}x{size}")
        glpk_times, smplx_times, phase2_times = [], [], []
        smplx_iters, phase2_iters = [], []

        for _ in range(n_runs):
            A = np.random.randint(0, 5, size=(size, size))
            b = np.random.randint(10, 100, size=size)
            c = np.random.randint(1, 20, size=size)

            # GLPK
            start = time.time()
            solve_glpk(A, b, c)
            glpk_times.append(time.time() - start)

            # Simplex estándar
            start = time.time()
            _, iters_s = simplex_standard(A, b, c)
            smplx_times.append(time.time() - start)
            smplx_iters.append(iters_s)

            # Simplex 2 fases
            start = time.time()
            _, iters2 = simplex_two_phase(A, b, c)
            phase2_times.append(time.time() - start)
            phase2_iters.append(iters2)

        results.append({
            "size": size,
            "glpk_time": np.mean(glpk_times),
            "simplex_time": np.mean(smplx_times),
            "two_phase_time": np.mean(phase2_times),
            "simplex_iters": np.mean(smplx_iters),
            "two_phase_iters": np.mean(phase2_iters)
        })

    return results

# ==== GRÁFICAS ====
def plot_results(results):
    sizes = [r["size"] for r in results]

    # === Gráfico 1: Tiempo de ejecución ===
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [r["glpk_time"] for r in results], label="GLPK (Pyomo)")
    plt.plot(sizes, [r["simplex_time"] for r in results], label="Simplex Estándar")
    plt.plot(sizes, [r["two_phase_time"] for r in results], label="Simplex Dos Fases")
    plt.xlabel("Tamaño del problema (n variables)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación de Tiempo de Ejecución")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/tiempo_ejecucion.png")
    plt.show()

    # === Gráfico 2: Número de iteraciones ===
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, [r["simplex_iters"] for r in results], label="Simplex Estándar")
    plt.plot(sizes, [r["two_phase_iters"] for r in results], label="Simplex Dos Fases")
    plt.xlabel("Tamaño del problema (n variables)")
    plt.ylabel("Número de iteraciones")
    plt.title("Comparación de Número de Iteraciones")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/iteraciones_simplex.png")
    plt.show()


# ==== EJECUCIÓN ====
if __name__ == "__main__":
    tamanios = list(range(5, 26, 5))  # 5, 10, 15, 20, 25
    resultados = benchmark(tamanios, n_runs=5)
    plot_results(resultados)
