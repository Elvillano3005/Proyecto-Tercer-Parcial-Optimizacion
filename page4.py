import streamlit as st 
import math
import numpy as np
import matplotlib.pyplot as plt
st.title("2. Bounding Phase Method (Método de fase de acotamiento)")
st.text("El método de fase de acotamiento se utiliza para acotar el mínimo de una función. Este método garantiza acotar el mínimo de una función unimodal. El algoritmo comienza con una suposición inicial y luego encuentra una dirección de búsqueda basada en dos evaluaciones más de la función en las proximidades de la suposición inicial. Posteriormente, se adopta una estrategia de búsqueda exponencial para alcanzar el óptimo.")
st.text("Algoritmo")
st.image("p4.png",caption="")

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


def fase_de_acotamiento(f, x0, delta, n):
    puntos_x = [x0]
    puntos_y = [f(x0)]

    f0 = f(x0)
    f1 = f(x0 + delta)
    puntos_x.append(x0 + delta)
    puntos_y.append(f1)

    if f1 > f0:
        delta = -delta
        f1 = f(x0 + delta)
        puntos_x.append(x0 + delta)
        puntos_y.append(f1)

        if f1 >= f0:
            return x0 + delta, x0 - delta, puntos_x, puntos_y

    x_prev = x0
    x_curr = x0 + delta

    for i in range(n):
        x_next = x_curr + (2 ** i) * delta
        f_next = f(x_next)
        puntos_x.append(x_next)
        puntos_y.append(f_next)

        if f_next >= f(x_curr):
            return x_prev, x_next, puntos_x, puntos_y

        x_prev = x_curr
        x_curr = x_next

    return x_prev, x_curr, puntos_x, puntos_y


def graficar_fase_de_acotamiento(a: float, b: float, f: callable, nombre: str, x0: float, delta: float, n: int):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [f(x) for x in x_vals]

    xmin, xmax, puntos_x, puntos_y = fase_de_acotamiento(f, x0, delta, n)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=xmin, color='green', linestyle='--', label='Inicio Intervalo')
    ax.axvline(x=xmax, color='orange', linestyle='--', label='Fin Intervalo')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{nombre} - Método de fase de acotamiento')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


st.title("Imprementacion")

funciones = {
    "Función lata": (lata, 0.2, 5),
    "Función caja": (caja, 2, 3),
    "Función 1": (funcion1, 0.01, 10),
    "Función 2": (funcion2, 0, 5),
    "Función 3": (funcion3, -2.5, 2.5),
    "Función 4": (funcion4, -1.5, 3),
    "Función 5": (funcion5, -5, 5)
}

opcion = st.selectbox("Selecciona una función para optimizar:", list(funciones.keys()))
funcion, a, b = funciones[opcion]

st.write("### Parámetros del método")
x0 = st.slider("x₀ (punto inicial)", float(a), float(b), float((a + b) / 2), step=0.01)
delta = st.slider("Δ (paso)", 0.0001, 1.0, 0.1, step=0.0001, format="%.4f")
n = st.slider("Número máximo de iteraciones", 1, 50, 10)

graficar_fase_de_acotamiento(a, b, funcion, opcion, x0, delta, n)