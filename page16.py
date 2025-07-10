import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
st.title("14.Random Walk")
st.text("Es un método aleatorio que se mueve sin considerar si el nuevo estado mejora o no la solución actual. Es útil para explorar el espacio de soluciones pero no garantiza mejoras ni convergencia.")
st.title("Algoritmo")
st.image("p16.png")
st.title("Implementacion")
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

def ackley(X):
    x, y = X
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

def sphere(X):
    return sum([x ** 2 for x in X])

def rosenbrock(X):
    return sum([100 * (X[i + 1] - X[i] ** 2) ** 2 + (1 - X[i]) ** 2 for i in range(len(X) - 1)])

def beale(X):
    x, y = X
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

def booth(X):
    x, y = X
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def himmelblau(X):
    x, y = X
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def mccormick(X):
    x, y = X
    return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

# Diccionario de funciones
funciones = {
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Beale": beale,
    "Booth": booth,
    "Himmelblau": himmelblau,
    "McCormick": mccormick
}

# Algoritmo Random Walk
def random_walk(func, bounds, max_iter, step_size):
    current = np.random.uniform(bounds[0], bounds[1], size=2)
    best = current.copy()
    history = [current.copy()]

    for _ in range(max_iter):
        next_step = current + np.random.uniform(-step_size, step_size, size=2)
        if np.all(bounds[0] <= next_step) and np.all(next_step <= bounds[1]):
            current = next_step
            if func(current) < func(best):
                best = current
            history.append(current.copy())

    return best, history



funcion_seleccionada = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
max_iter = st.slider("Número de iteraciones", 100, 2000, 500)
step_size = st.slider("Tamaño del paso", 0.01, 1.0, 0.1)

funcion = funciones[funcion_seleccionada]
limites = [-5, 5] if funcion_seleccionada != "Beale" else [-4.5, 4.5]

# Ejecutar algoritmo
mejor_sol, recorrido = random_walk(funcion, limites, max_iter, step_size)
recorrido = np.array(recorrido)

# Mostrar resultados
st.write(f"📍 Mejor solución encontrada: {mejor_sol}")
st.write(f"🔻 Valor de la función: {funcion(mejor_sol):.6f}")

# Graficar recorrido
x = np.linspace(limites[0], limites[1], 400)
y = np.linspace(limites[0], limites[1], 400)
X, Y = np.meshgrid(x, y)
Z = np.array([funcion([i, j]) for i, j in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
ax.plot(recorrido[:, 0], recorrido[:, 1], 'w.-', label='Recorrido')
ax.plot(mejor_sol[0], mejor_sol[1], 'r*', markersize=12, label='Óptimo')
ax.set_title(f"Recorrido - {funcion_seleccionada}")
ax.legend()
fig.colorbar(contour)
st.pyplot(fig)