import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt 

st.title("3. Interval Halving Method (Intervalos por la mitad)")
st.text("Este método divide el intervalo de búsqueda en tres partes, evaluando la función en dos puntos equidistantes dentro del intervalo. En cada iteración, se descarta la sección menos prometedora, reduciendo progresivamente el espacio de búsqueda. Se basa en la suposición de que la función es unimodal en el intervalo analizado.")
st.image("2.6.png", caption="")
st.text("Algoritmo")
st.image("p5.png",caption="")

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


def intervalos_por_la_mitad(funcion, a, b, epsilon, max_iter=50):
    puntos_x = []
    puntos_y = []

    for i in range(max_iter):
        if abs(b - a) < epsilon:
            break

        x1 = (a + b) / 2 - epsilon
        x2 = (a + b) / 2 + epsilon

        f1 = funcion(x1)
        f2 = funcion(x2)

        puntos_x.extend([x1, x2])
        puntos_y.extend([f1, f2])

        if f1 < f2:
            b = x2
        else:
            a = x1

    x_min = (a + b) / 2
    return x_min, puntos_x, puntos_y


def graficar_intervalos_por_la_mitad(a: float, b: float, funcion: callable, nombre: str, epsilon: float, max_iter: int):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [funcion(x) for x in x_vals]

    x_min, puntos_x, puntos_y = intervalos_por_la_mitad(funcion, a, b, epsilon, max_iter)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=x_min, color='green', linestyle='--', label=f'Mínimo aproximado x={x_min:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{nombre} - Intervalos por la mitad')
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
epsilon = st.slider("Tolerancia ε", 0.0001, 1.0, 0.01, step=0.0001, format="%.4f")
max_iter = st.slider("Número máximo de iteraciones", 1, 100, 50)

graficar_intervalos_por_la_mitad(a, b, funcion, opcion, epsilon, max_iter)