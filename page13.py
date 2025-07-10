import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
st.title("13.Búsqueda Unidireccional")
st.text("Una búsqueda unidireccional es una búsqueda unidimensional efectuada mediante la comparaci ́on de valores funcionales sólo a lo largo de una dirección especificada. Usualmente, una búsqueda unidireccional se efectúa desde un punto x(t) y en una dirección especificada s(t) . Esto es, sólo se consideran en el proceso de búsqueda aquellos puntos que yacen sobre una línea (en un espacio N-dimensional, donde N es el número de variables de decisión del problema) que pasa a través del punto x(t) y orientada a lo largo de la dirección de búsqueda s(t) .")
st.image("13.png",caption="")
st.title("Imprementacion")
def funcion_objetivo(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))

# Búsqueda Unidireccional
def busqueda_unidireccional(x0, pasos, paso=0.01):
    puntos = [x0]
    for _ in range(pasos):
        nuevo_x = x0 + paso
        if funcion_objetivo(nuevo_x) > funcion_objetivo(x0):
            x0 = nuevo_x
            puntos.append(x0)
        else:
            break
    return puntos



x0 = st.slider("Punto inicial (x0)", -2.0, 2.0, 0.0)
pasos = st.slider("Número máximo de pasos", 10, 200, 50)
paso = st.slider("Tamaño del paso", 0.001, 0.1, 0.01)

# Ejecutar búsqueda
puntos = busqueda_unidireccional(x0, pasos, paso)

# Graficar función y trayectoria
x = np.linspace(-2, 2, 400)
y = funcion_objetivo(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='Función objetivo', color='black')
ax.plot(puntos, funcion_objetivo(np.array(puntos)), 'g.-', label='Búsqueda Unidireccional')
ax.set_title("Trayectoria de la búsqueda")
ax.legend()

st.pyplot(fig)