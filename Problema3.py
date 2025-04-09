import numpy as np
import pyomo.environ as pyo
import time
import pandas as pd
from tabulate import tabulate

# Datos del problema
c = [5, 8, 3, 7, 6, 9, 4, 10, 2, 11]
A = np.array([
    [1, 2, 1, 1, 0, 0, 3, 1, 2, 1],
    [2, 1, 0, 2, 1, 1, 0, 3, 1, 2],
    [1, 1, 2, 0, 2, 1, 1, 0, 3, 1],
    [0, 2, 1, 1, 1, 0, 2, 1, 1, 1],
    [2, 0, 1, 1, 1, 2, 1, 1, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 2, 1, 0, 1, 2, 1, 1, 0],
    [1, 0, 1, 2, 1, 0, 1, 2, 1, 1]
])
b = [50, 60, 55, 40, 45, 70, 65, 50]

def solve_with_glpk():
    model = pyo.ConcreteModel()
    model.n = pyo.RangeSet(0, 9)
    model.x = pyo.Var(model.n, domain=pyo.NonNegativeReals)

    model.obj = pyo.Objective(expr=sum(c[i] * model.x[i] for i in model.n), sense=pyo.maximize)

    model.constraints = pyo.ConstraintList()
    for j in range(8):
        model.constraints.add(expr=sum(A[j][i] * model.x[i] for i in range(10)) <= b[j])

    solver = pyo.SolverFactory('glpk')
    start = time.time()
    results = solver.solve(model)
    end = time.time()

    Z = pyo.value(model.obj)
    solution = [pyo.value(model.x[i]) for i in model.n]
    return solution, Z, end - start, None  # Iteraciones no accesibles

def simplex_method_large_scale():
    c_min = np.array([-ci for ci in c])
    m, n = A.shape
    A_aug = np.hstack((A, np.eye(m)))
    c_aug = np.concatenate((c_min, np.zeros(m)))
    b_np = np.array(b)

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[0, :-1] = c_aug
    tableau[1:, :-1] = A_aug
    tableau[1:, -1] = b_np

    basic_vars = list(range(n, n + m))
    iteration = 0
    start = time.time()
    while np.any(tableau[0, :-1] < 0):
        iteration += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf for i in range(1, m+1)]
        if all(r == np.inf for r in ratios):
            return None, None, time.time() - start, iteration
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col
    end = time.time()
    solution = np.zeros(n)
    for i, var in enumerate(basic_vars):
        if var < n:
            solution[var] = tableau[i+1, -1]
    Z = -tableau[0, -1]
    return solution, -Z, end - start, iteration

def simplex_two_phase():
    m, n = A.shape
    A_phase1 = np.hstack((A, np.eye(m)))
    c_phase1 = np.concatenate((np.zeros(n), np.ones(m)))
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[0, :-1] = -c_phase1
    tableau[1:, :-1] = A_phase1
    tableau[1:, -1] = b
    tableau[0, :] = -np.sum(tableau[1:, :], axis=0)
    basic_vars = list(range(n, n + m))

    start = time.time()
    iter1 = 0
    while np.any(tableau[0, :-1] < 0):
        iter1 += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf for i in range(1, m+1)]
        if all(r == np.inf for r in ratios):
            return None, None, time.time() - start, iter1
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col

    if abs(tableau[0, -1]) > 1e-5:
        return None, None, time.time() - start, iter1

    tableau = tableau[:, :n + m + 1]
    tableau[0, :] = 0
    tableau[0, :n] = -np.array(c)
    for i, var in enumerate(basic_vars):
        if var < n:
            tableau[0] += tableau[0, var] * tableau[i+1]

    iter2 = 0
    while np.any(tableau[0, :-1] < 0):
        iter2 += 1
        pivot_col = np.argmin(tableau[0, :-1])
        ratios = [tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf for i in range(1, m+1)]
        if all(r == np.inf for r in ratios):
            return None, None, time.time() - start, iter1 + iter2
        pivot_row = np.argmin(ratios) + 1
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_val
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        basic_vars[pivot_row - 1] = pivot_col

    end = time.time()
    solution = np.zeros(n)
    for i, var in enumerate(basic_vars):
        if var < n:
            solution[var] = tableau[i+1, -1]
    Z = tableau[0, -1]
    return solution, Z, end - start, iter1 + iter2

if __name__ == "__main__":
    sol1, z1, t1, i1 = solve_with_glpk()
    sol2, z2, t2, i2 = simplex_method_large_scale()
    sol3, z3, t3, i3 = simplex_two_phase()

    print("\n=== Comparación de Métodos ===")
    print(f"GLPK (Pyomo):       Z = {z1:.2f}, Tiempo = {t1:.4f} s, Iteraciones = {i1 if i1 is not None else 'N/A'}")
    print(f"Simplex propio:      Z = {z2:.2f}, Tiempo = {t2:.4f} s, Iteraciones = {i2}")
    print(f"Simplex Dos Fases:   Z = {z3:.2f}, Tiempo = {t3:.4f} s, Iteraciones = {i3}")
