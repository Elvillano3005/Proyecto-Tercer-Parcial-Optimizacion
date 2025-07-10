import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

st.title("15.Hooke-Jeeves")
st.text("The pattern search method works by creating a set of search directions iteratively. The created search directions should be such that they completely span the search space. In other words, they should be such that starting from any point in the search space any other point in the search space can be reached by traversing along these search directions only. In a N -dimensional problem, this requires at least N linearly independent search directions. For example, in a two-variable function, at least two search directions are required to go from any one point to any other point. Among many possible combinations of N search directions, some combinations may be able to reach the destination faster (with lesser iterations), and some may require more iterations. In the Hooke-Jeeves method, a combination of exploratory moves and heuristic pattern moves is made iteratively. An exploratory move is performed in the vicinity of the current point systematically to find the best point around the current point. Thereafter, two such points are used to make a pattern move. We describe each of these moves in the following paragraphs:")
st.text("Movimiento exploratorio")
st.image("15.png",caption="")
st.text("Movimiento de patron")
st.text("A new point is found by jumping from the current best point xc along a direction connecting the previous best point x(k−1) and the current base point x(k) as follows:")
st.image("155.png")
st.text("Algoritmo")
st.image("p15.png",caption="")
st.title("Impremntacion")

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

# Algoritmo Hooke-Jeeves
def hooke_jeeves(func, x0, step_size=0.5, alpha=2.0, tolerance=1e-6, max_iter=500):
    def exploratory_search(x, step):
        x_new = np.copy(x)
        for i in range(len(x)):
            f0 = func(x_new)
            x_temp = np.copy(x_new)
            x_temp[i] += step
            if func(x_temp) < f0:
                x_new[i] += step
            else:
                x_temp[i] = x_new[i] - step
                if func(x_temp) < f0:
                    x_new[i] -= step
        return x_new

    x_base = np.copy(x0)
    x_new = exploratory_search(x_base, step_size)
    history = [x_base.copy()]

    it = 0
    while np.linalg.norm(x_new - x_base) > tolerance and it < max_iter:
        x_pattern = x_new + alpha * (x_new - x_base)
        x_base = x_new
        x_explore = exploratory_search(x_pattern, step_size)
        if func(x_explore) < func(x_new):
            x_new = x_explore
        else:
            step_size *= 0.5
        history.append(x_new.copy())
        it += 1

    return x_new, history



funcion_seleccionada = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
step_size = st.slider("Tamaño inicial del paso", 0.01, 1.0, 0.5)
max_iter = st.slider("Iteraciones máximas", 100, 2000, 500)

funcion = funciones[funcion_seleccionada]
limites = [-5, 5] if funcion_seleccionada != "Beale" else [-4.5, 4.5]
x0 = np.random.uniform(limites[0], limites[1], size=2)

# Ejecutar
mejor, recorrido = hooke_jeeves(funcion, x0, step_size=step_size, max_iter=max_iter)
recorrido = np.array(recorrido)

st.write(f"📍 Mejor solución: {mejor}")
st.write(f"🔻 Valor de la función: {funcion(mejor):.6f}")

# Gráfica
x = np.linspace(limites[0], limites[1], 400)
y = np.linspace(limites[0], limites[1], 400)
X, Y = np.meshgrid(x, y)
Z = np.array([funcion([i, j]) for i, j in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=50, cmap="plasma")
ax.plot(recorrido[:, 0], recorrido[:, 1], 'w.-', label='Recorrido')
ax.plot(mejor[0], mejor[1], 'r*', markersize=12, label='Óptimo')
ax.set_title(f"Hooke-Jeeves - {funcion_seleccionada}")
ax.legend()
fig.colorbar(contour)
st.pyplot(fig)