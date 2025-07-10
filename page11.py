import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

st.title("11.Método de la secante")
st.text("In the secant method, both magnitude and sign of derivatives are used to create a new point. The derivative of the function is assumed to vary linearly between the two chosen boundary points. Since boundary points have derivatives with opposite signs and the derivatives vary linearly between the boundary points, there exists a point between these two points with a zero derivative. Knowing the derivatives at the boundary points, the point with zero derivative can be easily found. If at two points x1 and x2, the quantity f′ (x1)f′ (x2) ≤0, the linear approximation of the derivative x1 and x2 will have a zero derivative at the point z given by")
st.image("11.png",caption="")
st.text("In this method, in one iteration more than half the search space may be eliminated depending on the gradient values at the two chosen points.However, smaller than half the search space may also be eliminated in one iteration.")
st.text("Algoritmo")
st.image("p11.png",caption="")

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


def secante(f_deriv, x0, x1, tol=1e-5, max_iter=100):
    puntos_x = []
    for _ in range(max_iter):
        f0 = f_deriv(x0)
        f1 = f_deriv(x1)
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        puntos_x.append(x2)
        if abs(x2 - x1) < tol:
            return x2, puntos_x
        x0, x1 = x1, x2
    return x1, puntos_x


def graficar_secante(f, f_deriv, x0, x1, nombre, tol):
    x_vals = np.linspace(min(x0, x1) - 1, max(x0, x1) + 1, 1000)
    y_vals = [f(x) for x in x_vals]

    x_opt, puntos_x = secante(f_deriv, x0, x1, tol)
    puntos_y = [f(x) for x in puntos_x]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=nombre)
    ax.scatter(puntos_x, puntos_y, color='red', s=10, label='Iteraciones')
    ax.axvline(x=x_opt, color='green', linestyle='--', label=f'Óptimo en x ≈ {x_opt:.4f}')
    ax.set_title(f"{nombre} - Método de la Secante")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


st.title("Imprementacion")

funciones = {
    "Función lata": (lata, lata_derivada, 0.5, 1.5),
    "Función caja": (caja, caja_derivada, 2.0, 2.5),
    "Función 1": (funcion1, funcion1_derivada, 1.0, 3.0),
    "Función 2": (funcion2, funcion2_derivada, -1.0, 1.0),
    "Función 3": (funcion3, funcion3_derivada, -2.0, 1.5),
    "Función 4": (funcion4, funcion4_derivada, -1.0, 2.0),
    "Función 5": (funcion5, funcion5_derivada, 0.0, 3.0),
}

opcion = st.selectbox("Selecciona una función:", list(funciones.keys()))
f, f_deriv, x0, x1 = funciones[opcion]

tolerancia = st.slider("Tolerancia", 1e-6, 1e-2, 1e-4, format="%.1e")

graficar_secante(f, f_deriv, x0, x1, opcion, tolerancia)