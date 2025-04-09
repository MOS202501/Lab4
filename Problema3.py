import numpy as np
import pyomo.environ as pyo
import time
import pandas as pd

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

# Modelo Pyomo
model = pyo.ConcreteModel()
model.n = pyo.RangeSet(0, 9)
model.x = pyo.Var(model.n, domain=pyo.NonNegativeReals)

# Función objetivo
model.obj = pyo.Objective(expr=sum(c[i] * model.x[i] for i in model.n), sense=pyo.maximize)

# Restricciones
model.constraints = pyo.ConstraintList()
for j in range(8):
    model.constraints.add(expr=sum(A[j][i] * model.x[i] for i in range(10)) <= b[j])

# Resolver con GLPK
solver = pyo.SolverFactory('glpk')
start = time.time()
results = solver.solve(model, tee=True)
end = time.time()

# Resultados
print("\n=== Resultados con GLPK ===")
for i in model.n:
    print(f"x{i+1} = {pyo.value(model.x[i]):.2f}")
print(f"Valor óptimo de Z: {pyo.value(model.obj):.2f}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")

import numpy as np
import time
from tabulate import tabulate

def create_tableau(A, b, c):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[0, :-1] = c
    tableau[1:, :-1] = A
    tableau[1:, -1] = b
    return tableau

def is_optimal(tableau):
    return np.all(tableau[0, :-1] >= 0)

def find_pivot_column(tableau):
    return np.argmin(tableau[0, :-1])

def find_pivot_row(tableau, pivot_col):
    ratios = []
    for i in range(1, tableau.shape[0]):
        if tableau[i, pivot_col] <= 0:
            ratios.append(float('inf'))
        else:
            ratios.append(tableau[i, -1] / tableau[i, pivot_col])
    if all(r == float('inf') for r in ratios):
        return None
    return np.argmin(ratios) + 1

def pivot_operation(tableau, pivot_row, pivot_col):
    new_tableau = tableau.copy()
    pivot_value = tableau[pivot_row, pivot_col]
    new_tableau[pivot_row] = tableau[pivot_row] / pivot_value
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            multiplier = tableau[i, pivot_col]
            new_tableau[i] = tableau[i] - multiplier * new_tableau[pivot_row]
    return new_tableau

def print_tableau(tableau, basic_vars):
    headers = [f"x{j+1}" for j in range(tableau.shape[1] - 1)] + ["RHS"]
    rows = [["Z"] + list(tableau[0])]
    for i in range(1, tableau.shape[0]):
        var = basic_vars[i-1]
        var_name = f"x{var+1}"
        rows.append([var_name] + list(tableau[i]))
    print(tabulate(rows, headers=headers, floatfmt=".2f"))

def extract_solution(tableau, basic_vars, n_variables):
    solution = np.zeros(n_variables)
    for i, var in enumerate(basic_vars):
        if var < n_variables:
            solution[var] = tableau[i+1, -1]
    return solution

def simplex_method_large_scale():
    # Datos del problema
    c = np.array([-5, -8, -3, -7, -6, -9, -4, -10, -2, -11])  # Maximizar -> minimizar negando
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
    b = np.array([50, 60, 55, 40, 45, 70, 65, 50])

    m, n = A.shape

    # Añadir variables de holgura
    A_aug = np.hstack((A, np.eye(m)))
    c_aug = np.concatenate((c, np.zeros(m)))
    basic_vars = list(range(n, n + m))

    # Crear el tableau
    tableau = create_tableau(A_aug, b, c_aug)

    # Tiempo de inicio
    start = time.time()

    # Iteraciones del método simplex
    iteration = 0
    while not is_optimal(tableau):
        iteration += 1
        print(f"\nIteración {iteration}")
        print_tableau(tableau, basic_vars)

        pivot_col = find_pivot_column(tableau)
        pivot_row = find_pivot_row(tableau, pivot_col)

        if pivot_row is None:
            print("El problema no tiene solución acotada.")
            return

        basic_vars[pivot_row - 1] = pivot_col
        tableau = pivot_operation(tableau, pivot_row, pivot_col)

    # Tiempo de fin
    end = time.time()

    print("\nTabla final (óptima):")
    print_tableau(tableau, basic_vars)

    # Extraer solución
    solution = extract_solution(tableau, basic_vars, n)
    Z = -tableau[0, -1]  # Negar porque habíamos convertido a minimización

    # Resultados
    print("\n=== Resultados del Método Simplex ===")
    for i in range(n):
        print(f"x{i+1} = {solution[i]:.2f}")
    print(f"Valor óptimo de Z: {-Z:.2f}")
    print(f"Tiempo de ejecución: {end - start:.4f} segundos")

if __name__ == "__main__":
    simplex_method_large_scale()

import numpy as np
import time
from tabulate import tabulate

def create_phase1_tableau(A, b):
    m, n = A.shape
    A_phase1 = np.hstack((A, np.eye(m)))  # Agregar variables artificiales
    c_phase1 = np.concatenate((np.zeros(n), np.ones(m)))  # Función objetivo Fase I
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[0, :n+m] = -c_phase1  # Minimizar suma artificiales => max -suma
    tableau[1:, :n+m] = A_phase1
    tableau[1:, -1] = b
    basic_vars = list(range(n, n + m))  # Artificiales como básicas

    # Ajustar Z correctamente como combinación de las restricciones artificiales
    tableau[0, :] = -np.sum(tableau[1:, :], axis=0)

    return tableau, basic_vars, n, m


def find_pivot_column(tableau):
    return np.argmin(tableau[0, :-1])

def find_pivot_row(tableau, pivot_col):
    ratios = []
    for i in range(1, tableau.shape[0]):
        if tableau[i, pivot_col] <= 0:
            ratios.append(np.inf)
        else:
            ratios.append(tableau[i, -1] / tableau[i, pivot_col])
    if all(r == np.inf for r in ratios):
        return None
    return np.argmin(ratios) + 1

def pivot_operation(tableau, pivot_row, pivot_col):
    tableau = tableau.copy()
    pivot_val = tableau[pivot_row, pivot_col]
    tableau[pivot_row] /= pivot_val
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
    return tableau

def print_tableau(tableau, basic_vars):
    headers = [f"x{j+1}" for j in range(tableau.shape[1] - 1)] + ["RHS"]
    rows = [["Z"] + list(tableau[0])]
    for i in range(1, tableau.shape[0]):
        var = basic_vars[i-1]
        var_name = f"x{var+1}"
        rows.append([var_name] + list(tableau[i]))
    print(tabulate(rows, headers=headers, floatfmt=".2f"))

def extract_solution(tableau, basic_vars, n):
    solution = np.zeros(n)
    for i, var in enumerate(basic_vars):
        if var < n:
            solution[var] = tableau[i+1, -1]
    return solution

def simplex_two_phase():
    # Datos del problema
    c = np.array([5, 8, 3, 7, 6, 9, 4, 10, 2, 11])
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
    b = np.array([50, 60, 55, 40, 45, 70, 65, 50])

    # Fase I
    tableau, basic_vars, n, m = create_phase1_tableau(A, b)
    print("\n=== Fase I: Minimización de variables artificiales ===")
    iter_f1 = 0
    while np.any(tableau[0, :-1] < 0):
        iter_f1 += 1
        print(f"\nIteración {iter_f1} - Fase I")
        print_tableau(tableau, basic_vars)
        pivot_col = find_pivot_column(tableau)
        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            print("Problema no acotado en Fase I.")
            return
        basic_vars[pivot_row - 1] = pivot_col
        tableau = pivot_operation(tableau, pivot_row, pivot_col)

    if abs(tableau[0, -1]) > 1e-5:
        print("\nEl problema no tiene solución factible.")
        return

    print("\n=== Fase I finalizada con solución factible ===")
    print_tableau(tableau, basic_vars)

    # Fase II
    print("\n=== Fase II: Maximizar función objetivo original ===")
    tableau = tableau[:, :n + m + 1]  # Eliminar columna extra si había
    tableau[0, :] = 0
    tableau[0, :n] = -c  # Convertir a -c para maximizar
    for i, var in enumerate(basic_vars):
        if var < n:
            tableau[0] += tableau[0, var] * tableau[i+1]

    iter_f2 = 0
    while np.any(tableau[0, :-1] < 0):
        iter_f2 += 1
        print(f"\nIteración {iter_f2} - Fase II")
        print_tableau(tableau, basic_vars)
        pivot_col = find_pivot_column(tableau)
        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            print("Problema no acotado en Fase II.")
            return
        basic_vars[pivot_row - 1] = pivot_col
        tableau = pivot_operation(tableau, pivot_row, pivot_col)

    # Resultado
    print("\n=== Tabla final (óptima) ===")
    print_tableau(tableau, basic_vars)

    solution = extract_solution(tableau, basic_vars, n)
    Z = tableau[0, -1]

    print("\n=== Solución óptima ===")
    for i in range(n):
        print(f"x{i+1} = {solution[i]:.2f}")
    print(f"Valor óptimo de Z: {Z:.2f}")

if __name__ == "__main__":
    start = time.time()
    simplex_two_phase()
    end = time.time()
    print(f"\nTiempo de ejecución: {end - start:.4f} segundos")
