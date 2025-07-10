import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt


st.title("8.Método de Newton-Raphson")
st.text("The goal of an unconstrained local optimization method is to achieve a point having as small a derivative as possible. In the Newton-Raphson method,a linear approximation to the first derivative of the function is made at apoint using the Taylor’s series expansion. That expression is equated to ze roto find the next guess. If the current point at iteration t is x(t), the point in the next iteration is governed by the following simple equation (obtained by considering up to the linear term in Taylor’s series expansion):")
st.text("Algoritmo")
st.image("p8.png",caption="")

def lata(r):
    return 2 * np.pi * r * r + (500 / r)

def lata_derivada(r):
    return 4 * np.pi * r - 500 / (r ** 2)

def lata_segunda(r):
    return 4 * np.pi + 1000 / (r ** 3)

def caja(l):
    return -(4 * l**3 - 60 * l**2 + 200 * l + 1)

def caja_derivada(l):
    return -(12 * l**2 - 120 * l + 200)

def caja_segunda(l):
    return -(24 * l - 120)

def funcion1(x):
    return float('inf') if x == 0 else x**2 + 54 / x

def funcion1_derivada(x):
    return 2 * x - 54 / x**2

def funcion1_segunda(x):
    return 2 + 108 / x**3

def funcion2(x):
    return x**3 + 2 * x - 3

def funcion2_derivada(x):
    return 3 * x**2 + 2

def funcion2_segunda(x):
    return 6 * x

def funcion3(x):
    return x**4 + x**2 - 33

def funcion3_derivada(x):
    return 4 * x**3 + 2 * x

def funcion3_segunda(x):
    return 12 * x**2 + 2

def funcion4(x):
    return 3 * x**4 - 8 * x**3 - 6 * x**2 + 12 * x

def funcion4_derivada(x):
    return 12 * x**3 - 24 * x**2 - 12 * x + 12

def funcion4_segunda(x):
    return 36 * x**2 - 48 * x - 12

def funcion5(x):
    return np.sin(x) + np.cos(2 * x)

def funcion5_derivada(x):
    return np.cos(x) - 2 * np.sin(2 * x)

def funcion5_segunda(x):
    return -np.sin(x) - 4 * np.cos(2 * x)


def newton_raphson_numerico(f, f1, f2, x0, tol=1e-5, max_iter=100):
    puntos_x = [x0]
    puntos_y = [f(x0)]

    for _ in range(max_iter):
        derivada = f1(x0)
        segunda = f2(x0)

        if segunda == 0:
            break  

        x1 = x0 - derivada / segunda

        puntos_x.append(x1)
        puntos_y.append(f(x1))

        if abs(x1 - x0) < tol:
            break

        x0 = x1

    return x1, puntos_x, puntos_y


def graficar_newton(f, f1, f2, a, b, x0, nombre, tol):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [f(x) for x in x_vals]

    x_min, puntos_x, puntos_y = newton_raphson_numerico(f, f1, f2, x0, tol)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=x_min, color='green', linestyle='--', label=f'Mínimo en x ≈ {x_min:.4f}')
    ax.set_title(f"{nombre} - Newton-Raphson")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


st.title("Imprementacion")

funciones = {
    "Función lata": (lata, lata_derivada, lata_segunda, 1.0, 0.2, 5),
    "Función caja": (caja, caja_derivada, caja_segunda, 2.5, 2, 3),
    "Función 1": (funcion1, funcion1_derivada, funcion1_segunda, 2.0, 0.01, 10),
    "Función 2": (funcion2, funcion2_derivada, funcion2_segunda, 1.0, 0, 5),
    "Función 3": (funcion3, funcion3_derivada, funcion3_segunda, 0.0, -2.5, 2.5),
    "Función 4": (funcion4, funcion4_derivada, funcion4_segunda, 1.0, -1.5, 3),
    "Función 5": (funcion5, funcion5_derivada, funcion5_segunda, 1.0, -5, 5)
}

opcion = st.selectbox("Selecciona una función para optimizar:", list(funciones.keys()))

f, f1, f2, x0, a, b = funciones[opcion]
tolerancia = st.slider("Tolerancia", 1e-6, 1e-2, 1e-4, format="%.1e")

graficar_newton(f, f1, f2, a, b, x0, opcion, tolerancia)