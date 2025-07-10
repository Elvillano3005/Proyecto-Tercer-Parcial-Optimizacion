import streamlit as st 
import math
import numpy as np
import matplotlib.pyplot as plt

st.title("5. Golden Section Search Method")
st.text("Este método es una variante optimizada del método de búsqueda de Fibonacci, en la que los puntos dentro del intervalo se eligen de acuerdo con la razón áurea. En cada iteración, se evalúa la función en dos puntos estratégicos y se descarta la región que no contiene el óptimo. Su principal ventaja es que reduce el número de evaluaciones necesarias, manteniendo una alta eficiencia en la búsqueda del mínimo o máximo.")
st.image("2.8.png", caption="")
st.text("Algortimo")
st.image("p7.png",caption="")

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

#def funcion5(x):
    return math.sin(x) + math.cos(2 * x)


def busqueda_dorada(funcion, a, b, tol=1e-5):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = funcion(x1)
    f2 = funcion(x2)

    puntos_x = [x1, x2]
    puntos_y = [f1, f2]

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = funcion(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = funcion(x2)

        puntos_x.extend([x1, x2])
        puntos_y.extend([f1, f2])

    x_min = (a + b) / 2
    return x_min, puntos_x, puntos_y


def graficar_dorada(a: float, b: float, funcion: callable, nombre: str, tolerancia: float):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [funcion(x) for x in x_vals]

    x_min, puntos_x, puntos_y = busqueda_dorada(funcion, a, b, tol=tolerancia)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=x_min, color='green', linestyle='--', label=f'Mínimo aprox x={x_min:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{nombre} - Búsqueda Dorada')
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
    #"Función 5 (sin(x) + cos(2x))": (funcion5, -5, 5)
}

opcion = st.selectbox("Selecciona una función para optimizar:", list(funciones.keys()))
funcion, a, b = funciones[opcion]

tolerancia = st.slider("Tolerancia (precisión)", 1e-6, 1e-2, 1e-4, format="%.1e")
graficar_dorada(a, b, funcion, opcion, tolerancia)