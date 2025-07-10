import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.title("14.Nelder-Mead Simplex")
st.text("El método Simplex también conocido como Nelder-Mead, fue propuesto por Jonh A. Nelder y Roger Mead en 1965. Este método requiere D + 1 puntos iniciales para funcionar (D es el número de variables del problema). Dichos puntos forman el Simplex inicial y debe cumplir con la siguiente característica: el Simplex no debe formar un hipercubo de volumen cero. Esto es, hablando de un problema de dos dimensiones, el Simplex no debe ser una línea recta, en tres dimensiones no deberá formar un plano, y en más dimensiones no debe formar un hiperplano.")
st.image("14.png",caption="")
st.text("Algoritmo")
st.image("p14.png",caption="")
st.title("imprementacion")
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



funcion_seleccionada = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
max_iter = st.slider("Iteraciones máximas", 100, 2000, 500)

funcion = funciones[funcion_seleccionada]
limites = [-5, 5] if funcion_seleccionada != "Beale" else [-4.5, 4.5]
x0 = np.random.uniform(limites[0], limites[1], size=2)

# Almacenar trayectoria manualmente
trayectoria = []

def callback(xk):
    trayectoria.append(np.copy(xk))

# Ejecutar Nelder-Mead
resultado = minimize(funcion, x0, method='Nelder-Mead', callback=callback, options={'maxiter': max_iter})
mejor = resultado.x
trayectoria = np.array(trayectoria)

st.write(f"📍 Mejor solución: {mejor}")
st.write(f"🔻 Valor de la función: {funcion(mejor):.6f}")

# Gráfica
x = np.linspace(limites[0], limites[1], 400)
y = np.linspace(limites[0], limites[1], 400)
X, Y = np.meshgrid(x, y)
Z = np.array([funcion([i, j]) for i, j in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=50, cmap="plasma")
if len(trayectoria) > 0:
    ax.plot(trayectoria[:, 0], trayectoria[:, 1], 'w.-', label='Trayectoria')
ax.plot(mejor[0], mejor[1], 'r*', markersize=12, label='Óptimo')
ax.set_title(f"Nelder-Mead - {funcion_seleccionada}")
ax.legend()
fig.colorbar(contour)
st.pyplot(fig)