import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Importar solvers profesionales
import pulp
from scipy.optimize import linprog

# Importar nuestras implementaciones del método Simplex
# (Usaremos las funciones necesarias que están en los archivos Problema1.py)
from Problema1 import create_tableau, is_optimal, find_pivot_column, find_pivot_row, pivot_operation, extract_solution


def nuestro_simplex(c, A, b, max_iter=1000):
    """
    Implementación de nuestro método Simplex para resolver problemas de programación lineal.

    Args:
        c: Coeficientes de la función objetivo (negados para maximización)
        A: Matriz de restricciones
        b: Lado derecho de las restricciones
        max_iter: Número máximo de iteraciones

    Returns:
        solution: Vector solución
        obj_value: Valor óptimo de la función objetivo
        time_taken: Tiempo de ejecución
        iterations: Número de iteraciones realizadas
    """
    start_time = time.time()

    # Preparar el problema
    m, n = A.shape

    # Añadir variables de holgura
    A_aug = np.hstack((A, np.eye(m)))
    c_aug = np.concatenate((c, np.zeros(m)))
    basic_vars = list(range(n, n + m))

    # Crear el tableau
    tableau = create_tableau(A_aug, b, c_aug)

    # Iteraciones del método simplex
    iteration = 0
    while not is_optimal(tableau) and iteration < max_iter:
        iteration += 1

        pivot_col = find_pivot_column(tableau)
        pivot_row = find_pivot_row(tableau, pivot_col)

        if pivot_row is None:
            return None, float('inf'), time.time() - start_time, iteration

        basic_vars[pivot_row - 1] = pivot_col
        tableau = pivot_operation(tableau, pivot_row, pivot_col)

    # Extraer solución
    solution = extract_solution(tableau, basic_vars)
    # Negar porque habíamos convertido a minimización
    obj_value = -tableau[0, -1]

    end_time = time.time()
    return solution[:n], obj_value, end_time - start_time, iteration


def pulp_solver(c, A, b, maximize=True):
    """
    Resolver un problema de programación lineal usando PuLP.

    Args:
        c: Coeficientes de la función objetivo
        A: Matriz de restricciones
        b: Lado derecho de las restricciones
        maximize: True si es un problema de maximización, False si es de minimización

    Returns:
        solution: Vector solución
        obj_value: Valor óptimo de la función objetivo
        time_taken: Tiempo de ejecución
    """
    start_time = time.time()

    m, n = A.shape

    # Crear el problema
    if maximize:
        prob = pulp.LpProblem("LP_Problem", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("LP_Problem", pulp.LpMinimize)

    # Crear variables
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]

    # Función objetivo
    prob += pulp.lpSum([c[i] * x[i] for i in range(n)]), "Objective Function"

    # Restricciones
    for i in range(m):
        prob += pulp.lpSum([A[i, j] * x[j]
                           for j in range(n)]) <= b[i], f"Constraint_{i+1}"

    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extraer solución
    solution = np.array([x[i].value() for i in range(n)])
    obj_value = pulp.value(prob.objective)

    end_time = time.time()
    return solution, obj_value, end_time - start_time


def scipy_solver(c, A, b, maximize=True):
    """
    Resolver un problema de programación lineal usando SciPy.

    Args:
        c: Coeficientes de la función objetivo
        A: Matriz de restricciones
        b: Lado derecho de las restricciones
        maximize: True si es un problema de maximización, False si es de minimización

    Returns:
        solution: Vector solución
        obj_value: Valor óptimo de la función objetivo
        time_taken: Tiempo de ejecución
    """
    start_time = time.time()

    # Ajustar para maximización si es necesario
    if maximize:
        c = -np.array(c)  # Negar para convertir maximización a minimización

    # Resolver usando SciPy
    result = linprog(c, A_ub=A, b_ub=b, method='simplex',
                     options={'disp': False})

    # Extraer solución
    solution = result.x
    obj_value = -result.fun if maximize else result.fun

    end_time = time.time()
    return solution, obj_value, end_time - start_time


def generar_problema_aleatorio(n_vars, n_restricciones, densidad=0.7, max_coef=10):
    """
    Generar un problema de programación lineal aleatorio.

    Args:
        n_vars: Número de variables
        n_restricciones: Número de restricciones
        densidad: Densidad de la matriz A (proporción de elementos no nulos)
        max_coef: Valor máximo para los coeficientes

    Returns:
        c: Coeficientes de la función objetivo
        A: Matriz de restricciones
        b: Lado derecho de las restricciones
    """
    # Generar coeficientes de la función objetivo
    c = np.random.randint(1, max_coef + 1, size=n_vars)

    # Generar matriz de restricciones con cierta densidad
    A = np.zeros((n_restricciones, n_vars))
    for i in range(n_restricciones):
        for j in range(n_vars):
            if np.random.random() < densidad:
                A[i, j] = np.random.randint(0, max_coef + 1)

    # Asegurar que cada restricción tenga al menos un coeficiente no nulo
    for i in range(n_restricciones):
        if np.sum(A[i, :]) == 0:
            A[i, np.random.randint(0, n_vars)] = np.random.randint(
                1, max_coef + 1)

    # Generar lado derecho de las restricciones
    # Generamos un punto factible aleatorio y calculamos b
    x_factible = np.random.randint(0, 5, size=n_vars)
    b = np.dot(A, x_factible) + np.random.randint(10, 30, size=n_restricciones)

    return c, A, b


def comparar_solvers(problemas):
    """
    Comparar el rendimiento de diferentes solvers en varios problemas.

    Args:
        problemas: Lista de tuplas (c, A, b, nombre) con los datos de cada problema

    Returns:
        resultados: DataFrame con los resultados de la comparación
    """
    resultados = []

    for c, A, b, nombre in problemas:
        print(f"\nResolviendo problema: {nombre}")
        print(
            f"Dimensiones: {A.shape[1]} variables, {A.shape[0]} restricciones")

        # Resolver con nuestro método Simplex
        try:
            sol_simplex, val_simplex, tiempo_simplex, iter_simplex = nuestro_simplex(
                np.negative(c), A, b)
            if sol_simplex is None:
                print("Nuestro Simplex: Problema no acotado o sin solución")
                simplex_status = "No resuelto"
            else:
                print(
                    f"Nuestro Simplex: Z = {val_simplex:.4f}, Tiempo = {tiempo_simplex:.6f}s, Iteraciones = {iter_simplex}")
                simplex_status = "Resuelto"
        except Exception as e:
            print(f"Error en nuestro Simplex: {e}")
            sol_simplex, val_simplex, tiempo_simplex, iter_simplex = None, None, float(
                'inf'), None
            simplex_status = "Error"

        # Resolver con PuLP
        try:
            sol_pulp, val_pulp, tiempo_pulp = pulp_solver(c, A, b)
            print(f"PuLP: Z = {val_pulp:.4f}, Tiempo = {tiempo_pulp:.6f}s")
            pulp_status = "Resuelto"
        except Exception as e:
            print(f"Error en PuLP: {e}")
            sol_pulp, val_pulp, tiempo_pulp = None, None, float('inf')
            pulp_status = "Error"

        # Resolver con SciPy
        try:
            sol_scipy, val_scipy, tiempo_scipy = scipy_solver(c, A, b)
            print(f"SciPy: Z = {val_scipy:.4f}, Tiempo = {tiempo_scipy:.6f}s")
            scipy_status = "Resuelto"
        except Exception as e:
            print(f"Error en SciPy: {e}")
            sol_scipy, val_scipy, tiempo_scipy = None, None, float('inf')
            scipy_status = "Error"

        # Calcular diferencia relativa en la solución (si ambos métodos encontraron solución)
        if val_simplex is not None and val_pulp is not None:
            diff_pulp = abs(val_simplex - val_pulp) / max(abs(val_pulp), 1e-10)
        else:
            diff_pulp = None

        if val_simplex is not None and val_scipy is not None:
            diff_scipy = abs(val_simplex - val_scipy) / \
                max(abs(val_scipy), 1e-10)
        else:
            diff_scipy = None

        # Guardar resultados
        resultados.append({
            'Problema': nombre,
            'Variables': A.shape[1],
            'Restricciones': A.shape[0],
            'Simplex_Status': simplex_status,
            'Simplex_Valor': val_simplex,
            'Simplex_Tiempo': tiempo_simplex,
            'Simplex_Iteraciones': iter_simplex,
            'PuLP_Status': pulp_status,
            'PuLP_Valor': val_pulp,
            'PuLP_Tiempo': tiempo_pulp,
            'SciPy_Status': scipy_status,
            'SciPy_Valor': val_scipy,
            'SciPy_Tiempo': tiempo_scipy,
            'Diff_PuLP': diff_pulp,
            'Diff_SciPy': diff_scipy
        })

    return pd.DataFrame(resultados)


def visualizar_resultados(resultados):
    """
    Visualizar los resultados de la comparación.

    Args:
        resultados: DataFrame con los resultados de la comparación
    """
    # Configuración de estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")

    # 1. Gráfico de tiempos de ejecución
    plt.figure(figsize=(12, 6))

    # Filtrar solo los problemas resueltos por todos los métodos
    df_completo = resultados[(resultados['Simplex_Status'] == 'Resuelto') &
                             (resultados['PuLP_Status'] == 'Resuelto') &
                             (resultados['SciPy_Status'] == 'Resuelto')].copy()

    if not df_completo.empty:
        # Preparar datos para gráfico
        df_melt = pd.melt(df_completo,
                          id_vars=['Problema', 'Variables', 'Restricciones'],
                          value_vars=['Simplex_Tiempo',
                                      'PuLP_Tiempo', 'SciPy_Tiempo'],
                          var_name='Método', value_name='Tiempo (s)')

        # Renombrar métodos para mejor visualización
        df_melt['Método'] = df_melt['Método'].replace({
            'Simplex_Tiempo': 'Nuestro Simplex',
            'PuLP_Tiempo': 'PuLP',
            'SciPy_Tiempo': 'SciPy'
        })

        # Ordenar por tamaño del problema
        df_melt['Tamaño'] = df_melt['Variables'] * df_melt['Restricciones']
        df_melt = df_melt.sort_values('Tamaño')

        # Crear gráfico de barras agrupadas
        ax = sns.barplot(x='Problema', y='Tiempo (s)',
                         hue='Método', data=df_melt)
        plt.title('Comparación de Tiempos de Ejecución', fontsize=14)
        plt.xlabel('Problema', fontsize=12)
        plt.ylabel('Tiempo (s)', fontsize=12)
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        plt.xticks(rotation=45)
        plt.legend(title='Método')
        plt.tight_layout()
        plt.savefig('results/comparacion_tiempos.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Gráfico de precisión (diferencia relativa)
        plt.figure(figsize=(12, 6))
        df_diff = pd.melt(df_completo,
                          id_vars=['Problema', 'Variables', 'Restricciones'],
                          value_vars=['Diff_PuLP', 'Diff_SciPy'],
                          var_name='Comparación', value_name='Diferencia Relativa')

        df_diff['Comparación'] = df_diff['Comparación'].replace({
            'Diff_PuLP': 'Simplex vs PuLP',
            'Diff_SciPy': 'Simplex vs SciPy'
        })

        # Ordenar por tamaño del problema
        df_diff['Tamaño'] = df_diff['Variables'] * df_diff['Restricciones']
        df_diff = df_diff.sort_values('Tamaño')

        ax = sns.barplot(x='Problema', y='Diferencia Relativa',
                         hue='Comparación', data=df_diff)
        plt.title('Diferencia Relativa en Soluciones', fontsize=14)
        plt.xlabel('Problema', fontsize=12)
        plt.ylabel('Diferencia Relativa', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Comparación')
        plt.tight_layout()
        plt.savefig('results/comparacion_precision.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Gráfico de escalabilidad
        plt.figure(figsize=(10, 6))

        # Crear una columna de tamaño del problema
        resultados['Tamaño'] = resultados['Variables'] * \
            resultados['Restricciones']

        # Filtrar solo los problemas resueltos por todos los métodos
        df_escala = resultados[(resultados['Simplex_Status'] == 'Resuelto') &
                               (resultados['PuLP_Status'] == 'Resuelto') &
                               (resultados['SciPy_Status'] == 'Resuelto')].copy()

        if not df_escala.empty:
            # Preparar datos para gráfico
            df_escala_melt = pd.melt(df_escala,
                                     id_vars=['Tamaño'],
                                     value_vars=['Simplex_Tiempo',
                                                 'PuLP_Tiempo', 'SciPy_Tiempo'],
                                     var_name='Método', value_name='Tiempo (s)')

            # Renombrar métodos para mejor visualización
            df_escala_melt['Método'] = df_escala_melt['Método'].replace({
                'Simplex_Tiempo': 'Nuestro Simplex',
                'PuLP_Tiempo': 'PuLP',
                'SciPy_Tiempo': 'SciPy'
            })

            # Crear gráfico de dispersión con líneas de tendencia
            sns.scatterplot(x='Tamaño', y='Tiempo (s)', hue='Método',
                            style='Método', s=100, data=df_escala_melt)

            # Añadir líneas de tendencia
            for metodo in df_escala_melt['Método'].unique():
                subset = df_escala_melt[df_escala_melt['Método'] == metodo]
                sns.regplot(x='Tamaño', y='Tiempo (s)', data=subset,
                            scatter=False, label=f'Tendencia {metodo}')

            plt.title(
                'Escalabilidad: Tiempo vs. Tamaño del Problema', fontsize=14)
            plt.xlabel(
                'Tamaño del Problema (Variables × Restricciones)', fontsize=12)
            plt.ylabel('Tiempo (s)', fontsize=12)
            plt.yscale('log')  # Escala logarítmica para mejor visualización
            plt.xscale('log')  # Escala logarítmica para mejor visualización
            plt.legend(title='Método')
            plt.tight_layout()
            plt.savefig('results/escalabilidad.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("No hay suficientes problemas resueltos por todos los métodos para generar gráficos comparativos.")


def main():
    # Definir problemas a resolver
    problemas = []

    # 1. Problema del Problema 1 (pequeño)
    c1 = np.array([3, 2, 5])
    A1 = np.array([
        [1, 1, 1],
        [2, 1, 1],
        [1, 4, 2]
    ])
    b1 = np.array([100, 150, 80])
    problemas.append((c1, A1, b1, "Problema 1 (pequeño)"))

    # 2. Problema del Problema 3 (mediano)
    c2 = np.array([5, 8, 3, 7, 6, 9, 4, 10, 2, 11])
    A2 = np.array([
        [1, 2, 1, 1, 0, 0, 3, 1, 2, 1],
        [2, 1, 0, 2, 1, 1, 0, 3, 1, 2],
        [1, 1, 2, 0, 2, 1, 1, 0, 3, 1],
        [0, 2, 1, 1, 1, 0, 2, 1, 1, 1],
        [2, 0, 1, 1, 1, 2, 1, 1, 0, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 1, 0, 1, 2, 1, 1, 0],
        [1, 0, 1, 2, 1, 0, 1, 2, 1, 1]
    ])
    b2 = np.array([50, 60, 55, 40, 45, 70, 65, 50])
    problemas.append((c2, A2, b2, "Problema 3 (mediano)"))

    # 3. Generar problemas aleatorios de diferentes tamaños
    np.random.seed(42)  # Para reproducibilidad

    # Problema pequeño aleatorio (5 variables, 3 restricciones)
    c3, A3, b3 = generar_problema_aleatorio(5, 3)
    problemas.append((c3, A3, b3, "Aleatorio pequeño (5×3)"))

    # Problema mediano aleatorio (10 variables, 8 restricciones)
    c4, A4, b4 = generar_problema_aleatorio(10, 8)
    problemas.append((c4, A4, b4, "Aleatorio mediano (10×8)"))

    # Problema grande aleatorio (15 variables, 12 restricciones)
    c5, A5, b5 = generar_problema_aleatorio(15, 12)
    problemas.append((c5, A5, b5, "Aleatorio grande (15×12)"))

    # Problema muy grande aleatorio (20 variables, 15 restricciones)
    c6, A6, b6 = generar_problema_aleatorio(20, 15)
    problemas.append((c6, A6, b6, "Aleatorio muy grande (20×15)"))

    # Comparar solvers
    print("\n===== ANÁLISIS DE RENDIMIENTO COMPARATIVO =====\n")
    resultados = comparar_solvers(problemas)

    # Mostrar tabla de resultados
    print("\n===== TABLA COMPARATIVA DE RESULTADOS =====\n")
    tabla_resultados = resultados[['Problema', 'Variables', 'Restricciones',
                                   'Simplex_Valor', 'Simplex_Tiempo', 'Simplex_Iteraciones',
                                   'PuLP_Valor', 'PuLP_Tiempo',
                                   'SciPy_Valor', 'SciPy_Tiempo']].copy()

    # Formatear para mejor visualización
    tabla_resultados['Simplex_Valor'] = tabla_resultados['Simplex_Valor'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    tabla_resultados['Simplex_Tiempo'] = tabla_resultados['Simplex_Tiempo'].apply(
        lambda x: f"{x:.6f}" if pd.notnull(x) else "N/A")
    tabla_resultados['PuLP_Valor'] = tabla_resultados['PuLP_Valor'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    tabla_resultados['PuLP_Tiempo'] = tabla_resultados['PuLP_Tiempo'].apply(
        lambda x: f"{x:.6f}" if pd.notnull(x) else "N/A")
    tabla_resultados['SciPy_Valor'] = tabla_resultados['SciPy_Valor'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    tabla_resultados['SciPy_Tiempo'] = tabla_resultados['SciPy_Tiempo'].apply(
        lambda x: f"{x:.6f}" if pd.notnull(x) else "N/A")

    # Renombrar columnas para mejor visualización
    tabla_resultados.columns = ['Problema', 'Variables', 'Restricciones',
                                'Valor (Simplex)', 'Tiempo (Simplex)', 'Iteraciones (Simplex)',
                                'Valor (PuLP)', 'Tiempo (PuLP)',
                                'Valor (SciPy)', 'Tiempo (SciPy)']

    print(tabulate(tabla_resultados, headers='keys',
          tablefmt='grid', showindex=False))

    # Visualizar resultados
    visualizar_resultados(resultados)

    print("\n===== CONCLUSIONES =====\n")
    print("1. Comparación de tiempos de ejecución:")
    print("   - Los solvers profesionales (PuLP y SciPy) suelen ser más rápidos que nuestra implementación")
    print("   - La diferencia de rendimiento aumenta con el tamaño del problema")
    print("   - Nuestra implementación es competitiva para problemas pequeños")

    print("\n2. Precisión de las soluciones:")
    print("   - Las soluciones encontradas por los diferentes métodos son generalmente similares")
    print("   - Pequeñas diferencias pueden deberse a la precisión numérica y criterios de parada")

    print("\n3. Escalabilidad:")
    print("   - Los solvers profesionales escalan mejor con el tamaño del problema")
    print("   - Nuestra implementación muestra un crecimiento más pronunciado en tiempo de ejecución")
    print("   - Para problemas muy grandes, la diferencia puede ser de órdenes de magnitud")

    print("\n4. Robustez:")
    print("   - Los solvers profesionales manejan mejor casos especiales y problemas mal condicionados")
    print("   - Nuestra implementación puede fallar en ciertos casos donde los solvers profesionales tienen éxito")

    print("\n5. Observaciones finales:")
    print("   - Los solvers profesionales incorporan décadas de optimizaciones y mejoras algorítmicas")
    print("   - Nuestra implementación sirve como herramienta educativa para entender el método Simplex")
    print("   - Para aplicaciones prácticas, es recomendable utilizar solvers profesionales")
    print("   - El desarrollo de algoritmos eficientes de optimización es un campo activo de investigación")


if __name__ == "__main__":
    main()
