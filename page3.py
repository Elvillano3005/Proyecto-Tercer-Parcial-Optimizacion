import streamlit as st 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("1. Exhaustive Search Method (Búsqueda Exhaustiva)")
st.text("Este método es el más simple, se encarga de encerrar el óptimo de una función calculando los valores de la función en un número de puntos espaciados equitativamente. Suele iniciar en el límite inferior de la variable de decisión, y tres valores de la función consecutivos son comparados cada vez, asumiendo la unimodalidad de la función.")
st.image("2.3.png", caption="")
st.text("Algoritmo")
st.image("p3.png",caption="")


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


def busqueda_exhaustiva(a: float, b: float, n: float, funcion: callable):
    delta_x = n
    x1 = a
    x2 = x1 + delta_x
    x3 = x2 + delta_x

    puntos_x = [x1, x2]
    puntos_y = [funcion(x1), funcion(x2)]

    while x3 <= b and not (funcion(x1) >= funcion(x2) <= funcion(x3)):
        x1 = x2
        x2 = x3
        x3 = x2 + delta_x

        puntos_x.append(x2)
        puntos_y.append(funcion(x2))

    return puntos_x, puntos_y


def graficar_funcion(a: float, b: float, funcion: callable, nombre: str, n: float):
    x = np.linspace(a, b, 1000)
    y = []
    for valor in x:
        try:
            y.append(funcion(valor))
        except:
            y.append(float('nan'))

    puntos_x, puntos_y = busqueda_exhaustiva(a, b, n, funcion)

    fig, ax = plt.subplots()
    ax.plot(x, y, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label=f'Puntos (n={n})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True)
    ax.set_title(nombre)
    st.pyplot(fig)


st.title("Implementacion")

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

n = st.slider("Selecciona el valor de n (paso de búsqueda)", min_value=0.0001, max_value=1.0, value=n_default, step=0.0001, format="%.4f")

graficar_funcion(a, b, funcion, opcion, n)