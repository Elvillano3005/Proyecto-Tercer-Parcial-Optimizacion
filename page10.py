import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

st.title("10.Método de Bisección")
st.text("The Newton-Raphson method involves computation of the second derivative, a numerical computation of which requires three function evaluations. In the bisection method, the computation of the second derivative is avoided; instead, only the first derivative is used. Both the function value and the sign of the first derivative at two points is used to eliminate a certain portion of the search space. This method is similar to the region-elimination methods discussed in Section 2.3.1, but in this method, derivatives are used to make the decision about the region to be eliminated. The algorithm once again assumes the unimodality of the function.")
st.text("Using the derivative information, the minimum is said to be bracketed in the interval (a, b) if two conditions—f′ (a) < 0 and f′ (b) > 0—are satisfied. Like other region-elimination methods, this algorithm also requires two initial boundary points bracketing the minimum. A bracketing algorithm described in Section 2.2 may be used to find the bracketing points. In the bisection method, derivatives at two boundary points and at the middle point are calculated and compared. Of the three points, two consecutive points with derivatives having opposite signs are chosen for the next iteration.")
st.text("Algoritmo")
st.image("p10.png",caption="")

def lata(r):
    return 2 * np.pi * r * r + (500 / r)

def lata_derivada(r):
    return 4 * np.pi * r - 500 / (r ** 2)

def caja(l):
    return -(4 * l**3 - 60 * l**2 + 200 * l + 1)

def caja_derivada(l):
    return -(12 * l**2 - 120 * l + 200)

def funcion1(x):
    return float('inf') if x == 0 else x**2 + 54 / x

def funcion1_derivada(x):
    return 2 * x - 54 / x**2

def funcion2(x):
    return x**3 + 2 * x - 3

def funcion2_derivada(x):
    return 3 * x**2 + 2

def funcion3(x):
    return x**4 + x**2 - 33

def funcion3_derivada(x):
    return 4 * x**3 + 2 * x

def funcion4(x):
    return 3 * x**4 - 8 * x**3 - 6 * x**2 + 12 * x

def funcion4_derivada(x):
    return 12 * x**3 - 24 * x**2 - 12 * x + 12

def funcion5(x):
    return np.sin(x) + np.cos(2 * x)

def funcion5_derivada(x):
    return np.cos(x) - 2 * np.sin(2 * x)


def biseccion(f_deriv, a, b, tol=1e-5, max_iter=100):
    puntos_x = []
    puntos_y = []

    for _ in range(max_iter):
        c = (a + b) / 2
        puntos_x.append(c)
        puntos_y.append(f_deriv(c))

        if abs(f_deriv(c)) < tol or (b - a) / 2 < tol:
            return c, puntos_x

        if f_deriv(a) * f_deriv(c) < 0:
            b = c
        else:
            a = c

    return c, puntos_x


def graficar_biseccion(f, f_deriv, a, b, nombre, tol):
    x_vals = np.linspace(a, b, 1000)
    y_vals = [f(x) for x in x_vals]

    x_opt, puntos_x = biseccion(f_deriv, a, b, tol)
    puntos_y = [f(x) for x in puntos_x]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Evaluaciones')
    ax.axvline(x=x_opt, color='green', linestyle='--', label=f'Óptimo en x ≈ {x_opt:.4f}')
    ax.set_title(f"{nombre} - Método de Bisección")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


st.title("Imprementacion")

funciones = {
    "Función lata": (lata, lata_derivada, 0.2, 5),
    "Función caja": (caja, caja_derivada, 2, 3),
    "Función 1": (funcion1, funcion1_derivada, 0.1, 10),
    "Función 2": (funcion2, funcion2_derivada, -2, 2),
    "Función 3": (funcion3, funcion3_derivada, -2.5, 2.5),
    "Función 4": (funcion4, funcion4_derivada, -1.5, 3),
    "Función 5": (funcion5, funcion5_derivada, -5, 5),
}

opcion = st.selectbox("Selecciona una función:", list(funciones.keys()))

f, f_deriv, a, b = funciones[opcion]
tolerancia = st.slider("Tolerancia", 1e-6, 1e-2, 1e-4, format="%.1e")

graficar_biseccion(f, f_deriv, a, b, opcion, tolerancia)