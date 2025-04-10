#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema 1 - Lab 4
Implementación del Método Simplex Estándar
"""

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import matplotlib.colors as colors


def simplex_method():

    # Coeficientes de la función objetivo (negados para maximización)
    c = np.array([-3, -2, -5, 0, 0, 0])

    # Matriz de restricciones con variables de holgura
    A = np.array([
        [1, 1, 1, 1, 0, 0],  # x1 + x2 + x3 + s1 = 100
        [2, 1, 1, 0, 1, 0],  # 2x1 + x2 + x3 + s2 = 150
        [1, 4, 2, 0, 0, 1]   # x1 + 4x2 + 2x3 + s3 = 80
    ])

    # Lado derecho de las restricciones
    b = np.array([100, 150, 80])

    # Variables básicas iniciales (variables de holgura)
    basic_vars = [3, 4, 5]  # s1, s2, s3 (índices 0-based)

    # Inicializar tabla simplex
    tableau = create_tableau(A, b, c)

    print("Problema de Programación Lineal:")
    print("Maximizar Z = 3x1 + 2x2 + 5x3")
    print("Sujeto a:")
    print("x1 + x2 + x3 ≤ 100")
    print("2x1 + x2 + x3 ≤ 150")
    print("x1 + 4x2 + 2x3 ≤ 80")
    print("x1, x2, x3 ≥ 0")
    print("\nForma estándar (con variables de holgura s1, s2, s3):")
    print("Maximizar Z = 3x1 + 2x2 + 5x3")
    print("Sujeto a:")
    print("x1 + x2 + x3 + s1 = 100")
    print("2x1 + x2 + x3 + s2 = 150")
    print("x1 + 4x2 + 2x3 + s3 = 80")
    print("x1, x2, x3, s1, s2, s3 ≥ 0")

    # Iterar hasta encontrar la solución óptima
    iteration = 0
    while not is_optimal(tableau):
        iteration += 1
        print(f"\nIteración {iteration}:")

        # Mostrar la tabla actual
        print_tableau(tableau, basic_vars)

        # Encontrar la columna pivote (variable de entrada)
        pivot_col = find_pivot_column(tableau)

        # Encontrar la fila pivote (variable de salida)
        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            print("El problema no tiene solución acotada.")
            return

        # Actualizar variables básicas
        # Ajustar el índice ya que pivot_row comienza desde 1
        basic_vars[pivot_row - 1] = pivot_col

        # Realizar la operación de pivoteo
        tableau = pivot_operation(tableau, pivot_row, pivot_col)

    # Mostrar la tabla final
    print("\nSolución óptima encontrada:")
    print_tableau(tableau, basic_vars)

    # Extraer la solución
    solution = extract_solution(tableau, basic_vars)

    # Mostrar la solución
    print("\nSolución óptima:")
    for i, val in enumerate(solution[:3]):
        print(f"x{i+1} = {val:.4f}")

    print(f"Valor óptimo de Z = {-tableau[0, -1]:.4f}")

    # Análisis de sensibilidad básico
    print("\nAnálisis de sensibilidad básico:")
    sensitivity_analysis(tableau, basic_vars)

    # Visualización geométrica
    plot_geometric_solution(A[:, :3], b, solution[:3])


def create_tableau(A, b, c):
    # Crear la tabla inicial del simplex
    m, n = A.shape  # m = número de restricciones, n = número de variables
    tableau = np.zeros((m+1, n+1))

    # Fila de costos
    tableau[0, :-1] = c

    # Matriz de restricciones
    tableau[1:, :-1] = A

    # Lado derecho
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
        return Non

    min_ratio_idx = np.argmin(ratios) + 1
    return min_ratio_idx


def pivot_operation(tableau, pivot_row, pivot_col):
    # Crear una copia del tableau
    new_tableau = tableau.copy()

    # Normalizar la fila pivote
    pivot_value = tableau[pivot_row, pivot_col]
    new_tableau[pivot_row] = tableau[pivot_row] / pivot_value

    # Actualizar las demás filas
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            multiplier = tableau[i, pivot_col]
            new_tableau[i] = tableau[i] - multiplier * new_tableau[pivot_row]

    return new_tableau


def print_tableau(tableau, basic_vars):
    headers = [f"x{j+1}" if j <
               3 else f"s{j-2}" for j in range(tableau.shape[1]-1)] + ["RHS"]

    # Preparar las filas con etiquetas
    rows = []
    rows.append(["Z"] + list(tableau[0]))

    for i in range(1, tableau.shape[0]):
        var_idx = basic_vars[i-1]
        var_name = f"x{var_idx+1}" if var_idx < 3 else f"s{var_idx-2}"
        rows.append([var_name] + list(tableau[i]))

    # Imprimir la tabla
    print(tabulate(rows, headers=headers, floatfmt=".4f"))


def extract_solution(tableau, basic_vars):
    # Extraer los valores de las variables en la solución óptima
    n_vars = tableau.shape[1] - 1
    solution = np.zeros(n_vars)

    # Obtener valores de variables básicas
    for i, var in enumerate(basic_vars):
        if i < len(tableau) - 1:  # Asegurarse de no exceder los límites
            solution[var] = tableau[i+1, -1]

    return solution


def plot_geometric_solution(A, b, solution):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generar puntos para graficar el plano
    x = np.linspace(0, max(b), 100)
    y = np.linspace(0, max(b), 100)
    X, Y = np.meshgrid(x, y)

    # Graficar cada restricción como un plano
    colors_planes = ['lightblue', 'lightgreen', 'lightpink']
    for i in range(A.shape[0]):
        # Despejar Z de la ecuación del plano
        Z = (b[i] - A[i, 0]*X - A[i, 1]*Y) / A[i, 2]
        Z = np.maximum(Z, 0)  # Restricción de no negatividad
        ax.plot_surface(X, Y, Z, alpha=0.3, color=colors_planes[i])

    # Graficar el punto óptimo
    ax.scatter(solution[0], solution[1], solution[2],
               color='red', s=100, label='Punto óptimo')

    # Configurar el gráfico
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('x₃')
    ax.set_title('Visualización Geométrica de la Solución')
    ax.legend()
    plt.savefig('results/problema_1_visualizacion_geometrica.png')


def sensitivity_analysis(tableau, basic_vars):
    # Análisis de sensibilidad para los coeficientes de la función objetivo
    n_vars = 3  # Solo analizamos las variables originales
    print("\nAnálisis detallado de sensibilidad:")
    print("\n1. Coeficientes de la función objetivo:")

    for j in range(n_vars):
        if j in basic_vars:
            idx = basic_vars.index(j)
            current_coef = -tableau[0, j]
            print(f"\nVariable x{j+1} (básica):")
            print(f"  - Coeficiente actual: {abs(current_coef):.4f}")
            print(f"  - Rango de estabilidad: (-∞, {tableau[0, j]:.4f})")
            print(f"  - Interpretación: La solución mantiene su estructura básica")
            print(f"    mientras el coeficiente no supere {tableau[0, j]:.4f}")
        else:
            print(f"\nVariable x{j+1} (no básica):")
            print(f"  - Coeficiente actual: {abs(tableau[0, j]):.4f}")
            print(f"  - Rango de estabilidad: ({-tableau[0, j]:.4f}, ∞)")
            print(f"  - Interpretación: La variable permanecerá en cero")
            print(
                f"    mientras el coeficiente no sea menor que {-tableau[0, j]:.4f}")

    # Análisis para los términos independientes
    print("\n2. Términos independientes (lado derecho):")
    for i in range(1, tableau.shape[0]):
        var_idx = basic_vars[i-1]
        var_name = f"x{var_idx+1}" if var_idx < 3 else f"s{var_idx-2}"
        current_value = tableau[i, -1]
        print(f"\nRestricción {i}:")
        print(f"  - Valor actual: {current_value:.4f}")
        print(f"  - Variable básica asociada: {var_name}")
        print(f"  - Efecto del cambio: Un incremento/decremento de Δ en el término")
        print(f"    independiente causará un cambio de ±Δ en {var_name}")
        if var_idx < 3:  # Si es una variable original
            print(
                f"  - Rango sugerido de variación: [{max(0, current_value-10):.4f}, {current_value+10:.4f}]")
            print(f"    para mantener la factibilidad de la solución")


def main():
    try:
        simplex_method()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
