# App en python que envía datos a un dashboard de streamlit

import streamlit as st
import pandas as pd

st.title('Nuestro primer dashboard')
st.write('Con un textito debajo')
st.write('## Un título de segundo nivel')
st.write('Texto en _cursiva_ y enlaces [larioja.org](https://larioja.org)')

# Lista con bullets
st.write("""- Arroz
- Pan
- Periodico""")

# Lista numerada
st.write("""1. Arroz
2. Pan
3. Periodico""")

# DataFrame (creamos una función para cargar cuando queramos)
@st.cache(ttl=3600) #Cacheamos la función de carga durante una hora
def load_data():
    df = pd.read_excel('https://actualidad.larioja.org/files/covid/6-2-incidencia-acumulada-la-rioja.xlsx?1622215565690', engine='openpyxl')
    # Reordenamos el dataframe
    df = df.sort_values('FECHA')
    df = df.set_index('FECHA')
    return df

df = load_data()
st.write(df)

# Checkbox
ver_grafica = st.checkbox('Ver Gráfica')

# Dibujar Gráfica si está activado el checkbox
if ver_grafica:
    st.line_chart(df)

# Barra lateral
with st.sidebar:
    st.write('Hola')
    st.line_chart(df['IA 14 DIAS'])
    # Radio buttons
    status = st.radio('Select gender:', ('Male', 'Female'))
    if status == 'Male':
        st.success('Male')
    else:
        st.success('Female')

# Botones
if st.button('About'):
    st.text('Este curso mola mazo')
