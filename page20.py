import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad, hessian
import autograd.numpy as anp

st.title("18.Método de Cauchy")
st.text("Este método trabaja bien cuando x(0) se encuentra lejos del óptimo.Cuando el punto actual está muy cercano el cambio en el gradiente es pequeño. Esto hace que la convergencia se lenta, para acelerarla se puede usar las derivadas de segundo orden.")

st.title("Algoritmo")
st.image("p20.png")
st.title("Imprementacion")

def sphere(X):
    return anp.sum(X**2)

def rosenbrock(X):
    return anp.sum(100*(X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)

def beale(X):
    x, y = X
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def booth(X):
    x, y = X
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def himmelblau(X):
    x, y = X
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

funciones = {
    "Rosenbrock": rosenbrock,
    "Sphere": sphere,
    "Beale": beale,
    "Booth": booth,
    "Himmelblau": himmelblau
}

# Golden Section Search
def golden_section_search(phi, a=0, b=2, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if phi(c) < phi(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2



funcion_seleccionada = st.selectbox("Selecciona una función", list(funciones.keys()))
max_iter = st.slider("Máximo de iteraciones", 10, 1000, 200)
tolerancia = st.slider("Tolerancia", 1e-6, 1e-2, 1e-4, format="%.1e")

f = funciones[funcion_seleccionada]
grad_f = grad(f)

limites = [-5, 5] if funcion_seleccionada != "Beale" else [-4.5, 4.5]
xk = anp.random.uniform(limites[0], limites[1], 2)

trayectoria = [xk.copy()]

for i in range(max_iter):
    gk = grad_f(xk)
    dk = -gk

    # Definimos phi(α) = f(x + α*d)
    phi = lambda alpha: f(xk + alpha * dk)

    # Buscar α óptimo
    alpha_k = golden_section_search(phi)

    xk1 = xk + alpha_k * dk
    trayectoria.append(xk1.copy())

    if np.linalg.norm(xk1 - xk) < tolerancia:
        break

    xk = xk1

trayectoria = np.array(trayectoria)

st.write(f"📍 Solución encontrada: {xk}")
st.write(f"🔻 Valor de la función: {f(xk):.6f}")
st.write(f"🔁 Iteraciones realizadas: {len(trayectoria)}")

# Gráfica
x = np.linspace(limites[0], limites[1], 400)
y = np.linspace(limites[0], limites[1], 400)
X, Y = np.meshgrid(x, y)
Z = np.array([f(anp.array([i, j])) for i, j in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=50, cmap="plasma")
ax.plot(trayectoria[:, 0], trayectoria[:, 1], 'r.-', label='Trayectoria')
ax.plot(xk[0], xk[1], 'r*', markersize=12, label='Óptimo')
ax.set_title(f"Cauchy con Golden Search - {funcion_seleccionada}")
ax.legend()
fig.colorbar(contour)
st.pyplot(fig)