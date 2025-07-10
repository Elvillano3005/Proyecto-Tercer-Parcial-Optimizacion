import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Método de eliminación de regiones")
st.text("El método de eliminación de regiones es un algoritmo de optimización unidimensional que reduce iterativamente el intervalo de búsqueda donde se espera encontrar el óptimo (mínimo o máximo). En cada paso:.")
st.image("z.png", caption="")
st.title("Imprementacion")

def lata(r):
    try:
        return 2 * math.pi * r * r + (500 / r)
    except ZeroDivisionError:
        return float('inf')

def caja(l):
    return -(4 * pow(l, 3) - 60 * l * l + 200 * l + 1)

def funcion1(x):
    return float('inf') if x == 0 else x**2 + 54 / x

def funcion2(x):
    return x**3 + 2 * x - 3

def funcion3(x):
    return x**4 + x**2 - 33

def funcion4(x):
    return 3 * x**4 - 8 * x**3 - 6 * x**2 + 12 * x

def funcion5(x):
    return math.sin(x) + math.cos(2 * x)

def funcion_objetivo(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))

funciones = {
    "Función lata": (lata, 0.2, 5, 0.5),
    "Función caja": (caja, 2, 3, 0.1),
    "Función 1": (funcion1, 0.01, 10, 0.01),
    "Función 2": (funcion2, 0, 5, 0.01),
    "Función 3": (funcion3, -2.5, 2.5, 0.001),
    "Función 4": (funcion4, -1.5, 3, 0.001),
    "Función 5": (funcion5, -5, 5, 0.01)
}

opcion = st.selectbox("Selecciona una función para visualizar:", list(funciones.keys()))

funcion, a, b, n_default = funciones[opcion]


def eliminacion_regiones(a, b, tolerancia, max_iter):
    historial = []
    for _ in range(max_iter):
        x1 = a + (b - a) / 3
        x2 = b - (b - a) / 3

        f1 = funcion_objetivo(x1)
        f2 = funcion_objetivo(x2)
        historial.append((a, b))

        if abs(b - a) < tolerancia:
            break

        if f1 < f2:
            a = x1
        else:
            b = x2

    x_opt = (a + b) / 2
    return x_opt, historial



a = st.slider("Límite inferior (a)", -2.0, 1.5, -1.0)
b = st.slider("Límite superior (b)", a + 0.1, 2.0, 1.0)
tolerancia = st.slider("Tolerancia", 0.001, 0.1, 0.01)
max_iter = st.slider("Iteraciones máximas", 1, 100, 30)

x_opt, historial = eliminacion_regiones(a, b, tolerancia, max_iter)


x = np.linspace(-2, 2, 500)
y = funcion_objetivo(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="Función objetivo", color='black')

for (ai, bi) in historial:
    ax.axvspan(ai, bi, color='orange', alpha=0.1)

ax.axvline(x_opt, color='green', linestyle='--', label=f"Máximo estimado: x ≈ {x_opt:.3f}")
ax.set_title("Eliminación de Regiones ")
ax.legend()

st.pyplot(fig)