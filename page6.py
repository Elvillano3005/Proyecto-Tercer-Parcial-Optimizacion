import streamlit as st 
import math
import numpy as np
import matplotlib.pyplot as plt

st.title("4. Fibonacci Search Method")
st.text("Este método utiliza la secuencia de Fibonacci para seleccionar puntos dentro del intervalo de búsqueda de manera óptima. En cada iteración, se comparan dos puntos y se elimina la región menos prometedora, reduciendo el intervalo de búsqueda. La ventaja principal de este método es que minimiza el número de evaluaciones de la función, lo que lo hace más eficiente que otros métodos de búsqueda sin derivadas.")
st.image("2.7.png", caption="")
st.text("Algoritmo")
st.image("p6.png",caption="")

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


def generar_fibonacci(n):
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


def busqueda_fibonacci(funcion, a, b, n):
    fib = generar_fibonacci(n + 1)
    puntos_x = []
    puntos_y = []

    k = 1
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)
    f1 = funcion(x1)
    f2 = funcion(x2)
    
    puntos_x.extend([x1, x2])
    puntos_y.extend([f1, f2])

    while k < n - 1:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n - k - 1] / fib[n - k]) * (b - a)
            f2 = funcion(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n - k - 2] / fib[n - k]) * (b - a)
            f1 = funcion(x1)

        puntos_x.extend([x1, x2])
        puntos_y.extend([f1, f2])
        k += 1

    x_min = (x1 + x2) / 2
    return x_min, puntos_x, puntos_y


def graficar_fibonacci(a: float, b: float, funcion: callable, nombre: str, n: int):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [funcion(x) for x in x_vals]

    x_min, puntos_x, puntos_y = busqueda_fibonacci(funcion, a, b, n)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=x_min, color='green', linestyle='--', label=f'Mínimo aprox x={x_min:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{nombre} - Búsqueda de Fibonacci')
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

n = st.slider("Número de iteraciones (n ≥ 5)", 5, 30, 15)
graficar_fibonacci(a, b, funcion, opcion, n)