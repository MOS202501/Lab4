import numpy as np
import pandas as pd

class SimplexSolver:
    def __init__(self):
        # Definimos la matriz A del problema en forma estándar
        # x1, x2, x3 son las variables originales
        # e1 (col 3): variable de exceso para desigualdad ≥
        # s1 (col 4): variable de holgura para ≤
        # a1, a2 (cols 5 y 6): variables artificiales
        self.A = np.array([
            [ 2,  1, -1,  0, 1, 0, 0],   # Ecuación (22) transformada (==): necesita a1
            [ 1, -3,  2, -1, 0, 1, 0],   # Ecuación (23): ≥ se transforma con e1 y a2
            [ 1,  1,  1,  0, 0, 0, 1]    # Ecuación (24): ≤ se transforma con s1
        ], dtype=float)

        # Vector del lado derecho del sistema (términos independientes)
        self.b = np.array([10, 5, 15], dtype=float)

        # Convertimos la función objetivo original a maximización
        # Min Z = 5x1 - 4x2 + 3x3  → Max Z = -5x1 + 4x2 -3x3
        # Las variables adicionales (e1, s1, a1, a2) no participan en la función objetivo
        self.c = np.array([-5, 4, -3, 0, 0, 0, 0], dtype=float)

        self.num_vars = 3  # Variables originales: x1, x2, x3
        self.num_artificiales = [5, 6]  # Columnas de las variables artificiales: a1, a2

    def imprimir_tabla(self, tableau, z, iteracion):
        # Imprime el tableau actual del método Simplex en forma de tabla
        headers = [f"x{i+1}" for i in range(tableau.shape[1] - 1)] + ["b"]
        df = pd.DataFrame(tableau, columns=headers, index=[f"F{i+1}" for i in range(len(tableau))])
        print(f"\n--- Iteración {iteracion} ---")
        print(df)
        print(f"Z = {z:.2f}\n")

    def fase_I(self):
        print("== FASE I: Búsqueda de solución básica factible ==\n")

        # Extendemos A con la columna b
        tableau = np.hstack([self.A, self.b.reshape(-1, 1)])

        # Creamos la fila Z (función objetivo artificial): minimizar la suma de variables artificiales
        z = np.zeros(tableau.shape[1])
        for i in range(len(tableau)):
            # Restamos cada fila con una variable artificial activa (a1, a2)
            if 5 + i < tableau.shape[1] - 1:  # Solo si hay artificiales en la columna esperada
                z -= tableau[i]

        # Inicializamos la base con las artificiales (a1, a2) y s1
        basis = [5, 6, 4]  # Índices de columnas básicas

        iteracion = 0
        self.imprimir_tabla(tableau, z[-1], iteracion)

        # Bucle del método Simplex
        while True:
            # Verificamos optimalidad (todos los coeficientes ≤ 0)
            if np.all(z[:-1] >= -1e-8):
                break

            # Selección de columna pivote: el más negativo
            pivot_col = np.argmin(z[:-1])

            # Selección de fila pivote: prueba de razón mínima
            ratios = np.divide(tableau[:, -1], tableau[:, pivot_col], where=tableau[:, pivot_col] > 1e-8)
            ratios[tableau[:, pivot_col] <= 1e-8] = np.inf
            pivot_row = np.argmin(ratios)

            # Pivoteo: hacer el pivote 1 y el resto 0 en su columna
            pivot = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot
            for i in range(len(tableau)):
                if i != pivot_row:
                    tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

            # Actualizar la fila Z y la base
            z -= z[pivot_col] * tableau[pivot_row]
            basis[pivot_row] = pivot_col

            iteracion += 1
            self.imprimir_tabla(tableau, z[-1], iteracion)

        # Verificamos que no haya variables artificiales en la base al terminar
        for art_col in self.num_artificiales:
            if art_col in basis:
                raise ValueError("No se encontró solución básica factible. Problema infactible.")

        # Eliminamos columnas de artificiales antes de pasar a Fase II
        tableau = np.delete(tableau, self.num_artificiales, axis=1)
        new_basis = [b for b in basis if b not in self.num_artificiales]

        return tableau, new_basis

    def fase_II(self, tableau, basis):
        print("== FASE II: Resolución del problema original ==\n")

        # Usamos la función objetivo original para construir Z
        c = self.c[:tableau.shape[1] - 1]  # Recortamos c a las columnas que quedaron
        cb = c[basis]
        z = cb @ tableau[:, :-1]
        z0 = cb @ tableau[:, -1]
        z_row = np.hstack([z - c, z0])

        iteracion = 0
        self.imprimir_tabla(tableau, z0, iteracion)

        # Método simplex clásico
        while True:
            if np.all(z_row[:-1] <= 1e-8):
                break  # Óptimo alcanzado

            pivot_col = np.argmax(z_row[:-1])
            ratios = np.divide(tableau[:, -1], tableau[:, pivot_col], where=tableau[:, pivot_col] > 1e-8)
            ratios[tableau[:, pivot_col] <= 1e-8] = np.inf
            pivot_row = np.argmin(ratios)

            pivot = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot
            for i in range(len(tableau)):
                if i != pivot_row:
                    tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

            # Actualizamos base y Z
            basis[pivot_row] = pivot_col
            cb = c[basis]
            z = cb @ tableau[:, :-1]
            z0 = cb @ tableau[:, -1]
            z_row = np.hstack([z - c, z0])

            iteracion += 1
            self.imprimir_tabla(tableau, z0, iteracion)

        # Presentamos solución óptima
        print("\n== Solución óptima encontrada ==")
        solucion = np.zeros(self.num_vars)

        # Solo imprimimos variables originales
        for i, var in enumerate(basis):
            if var < self.num_vars:
                solucion[var] = tableau[i, -1]

        for i in range(self.num_vars):
            print(f"x{i+1} = {solucion[i]:.2f}")

        print(f"Valor óptimo de Z = {-z0:.2f}")

    def resolver(self):
        tableau, basis = self.fase_I()
        self.fase_II(tableau, basis)

# === EJECUCIÓN ===
if __name__ == "__main__":
    solver = SimplexSolver()
    solver.resolver()
