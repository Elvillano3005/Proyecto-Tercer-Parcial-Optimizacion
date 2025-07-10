import streamlit as st 


pg = st.navigation([
    st.Page("page1.py", title="Menú principal", icon=""),
    st.Page("page2.py", title="Método de eliminación de regiones", icon=""),
    st.Page("page3.py", title="Exhaustive Search Method", icon=""),
    st.Page("page4.py", title="Bounding Phase Method", icon=""),
    st.Page("page5.py", title="Interval Halving Method", icon=""),
    st.Page("page6.py", title="Fibonacci Search Method", icon=""),
    st.Page("page7.py", title="Golden Section Search Method", icon=""),
    st.Page("page8.py", title="Método de Newton-Raphson", icon=""),
    st.Page("page9.py", title="Central Difference Method", icon=""),
    st.Page("page10.py", title="Método de Bisección", icon=""),
    st.Page("page11.py", title="Método de la secante", icon=""),
    st.Page("page12.py", title="Funciones Multivariadas", icon=""),
    st.Page("page13.py", title="Búsqueda Unidireccional", icon=""),
    st.Page("page14.py", title="Nelder-Mead Simplex", icon=""),
    st.Page("page15.py", title="Hooke-Jeeves", icon=""),
    st.Page("page16.py", title="Random Walk", icon=""),
    st.Page("page17.py", title="Hill Climbing", icon=""),
    st.Page("page18.py", title="Metodo Recocido Simulado", icon=""),
    st.Page("page19.py", title="Metodo de  gradiante", icon=""),
    st.Page("page20.py", title="Metodo de Cauchy ", icon=""),
    st.Page("page21.py", title="Metodo newton", icon=""),
    
])
pg.run()


