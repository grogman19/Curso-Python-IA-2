# Para la segunda parte del proyecto, crearemos un dashboard interactivo usando streamlit, en el que podremos introducir los datos
# de edad, sexo, y clase, de un hipotético pasajero del Titanic, y mediante el clasificador entrenado en la primera parte, hacer
# una predicción sobre si el pasajero hubiera sobrevivido al hundimiento o no.

# Comenzaremos por importar el módulo streamlit, y el resto de módulos necesarios para todo lo que vamos a hacer
import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image

# Mostramos un título y una breve explicación de lo que hace el dashboard
st.title('Simulador de supervivencia del Titanic')
st.write('Bienvenido al simulador de supervencia al desastre del Titanic. Si desea sentir de primera mano lo que pudo ser la experiencia del hundimiento del Titanic, introduzca sus datos de Sexo, Edad, y en que clase le gustaría viajar, y gustosamente le indicaremos si sobrevivió al desastre... o no.')

# Crearemos un form de Streamlit, para recargar la app únicamente cuando le demos al botón de 
# calcular, y no al interactuar con los otros controles
form_datos_entrada = st.form(key='datos_entrada')
# Introducimos un combo box para elegir el Sexo
sexo = form_datos_entrada.selectbox('Seleccione su sexo:', ('Hombre', 'Mujer'))
# Introducimos un campo de texto libre para la Edad
edad = form_datos_entrada.text_input(label='Introduzca su edad:') 
# Introducimos un combo box para elegir la Clase
clase = form_datos_entrada.selectbox('Ud. desea viajar en clase:', ('Primera', 'Segunda', 'Tercera'))
# Creamos un checkbox para mostrar los datos adicionales
ver_datos = form_datos_entrada.checkbox('Ver datos adicionales sobre el Titanic (para comprender mejor por que has sobrevivido, o no)')
# Introducimos un botón que procederá a realizar el cálculo
boton_calculo = form_datos_entrada.form_submit_button(label='Calcular')

# Importamos los datos del clasificador para hacer la predicción
# Comenzamos abriendo el fichero serializado con Pickle
with open('titanic_model.pkl', 'rb') as f:
    info = pickle.load(f)
        
# Extraemos los datos del diccionario
knc_titanic = info['modelo']
sc_titanic = info['scaler']
feature_names = info['feature_names']
target_name = info['target_name']
categorias_sexo = info['categorias']['Sex']

# Creamos un diccionario donde guardaremos los datos recibidos del usuario
dato = dict()

# Cargamos un par de imagenes que mostraremos en función del resultado
img_ok = Image.open('OK.jpg')
img_ko = Image.open('KO.jpg')

# Cargamos los datos originales del manifiesto del Titanic, para mostrar datos adicionales si lo pide el usuario
# Cargamos los datos en un dataframe
df_titanic = pd.read_csv('titanic.csv')

    
# Esta sección solo se ejecuta al presionar el botón Calcular    
if boton_calculo:
    # Sacamos el dato del sexo
    if sexo == 'Hombre':
        dato['Sex'] = categorias_sexo['male']
    else:
        dato['Sex'] = categorias_sexo['female']
    
    # Sacamos el dato de la clase
    if clase == 'Primera':
        dato['Pclass'] = 1
    elif clase == 'Segunda':
        dato['Pclass'] = 2
    else:
        dato['Pclass'] = 3

    # Sacamos el dato de la edad llevando control de los errores
    int_exception = False
    try:
        dato['Age'] = int(edad)
    except ValueError:
        st.error('ERROR: La edad debe de ser un número entero')
        int_exception = True
    
    # Si no ha habido excepción con la conversión de la edad a entero, continuamos
    if int_exception == False:
        # Hacemos un poco el tonto con una progress bar y mensajes de status para hacer más simpatico el dashboard y
        # que parezca que hace algo más sofisticado
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text('Ajustando modelo calibrando la hipótesis nula')
        for i in range(100):
            progress_bar.progress(i + 1)
            if i == 24:
                status_text.text('Entrenando red neuronal con un billón de hiperparámetros')
            elif i == 49:
                status_text.text('Entrenando red GaN para optimizar el resultado')
            elif i == 74:
                status_text.text('Precalentando Tensor Cores de la GPU para generar la predicción')
            elif i == 99:
                status_text.text('Cálculo de la predicción completado')
            time.sleep(0.1)

        # Y ahora sí, procedemos a realizar la predicción
        # Creamos un dataframe de pandas con los datos para la predicción
        X = pd.DataFrame()
        for categoria in feature_names:
            X.loc[0, categoria] = dato[categoria]
        
        # Normalizamos los datos de entrada con el scaler
        X_std = sc_titanic.transform(X)

        # Realizamos la predicción
        Y = pd.DataFrame()
        Y[target_name] = knc_titanic.predict(X_std)

        # Mostramos el resultado en pantalla
        if Y.loc[0, target_name] == 0:
            st.title('Ooooohhhh, lo sentimos, pero no había suficiente sitio en la tabla de madera, y NO HAS SOBREVIVIDO :(')
            st.image(img_ko)
        else:
            st.title('¡¡¡Enhorabuena!!! Había sitio en la tabla de madera para ti, y HAS SOBREVIVIDO :)')
            st.image(img_ok)

# Parte opcional 1: Ver un análisis de los datos originales del Titanic

# Dibujar gráficas si está activado el checkbox
if ver_datos:
    # Comenzamos mostrando un texto explicativo de la gráfica
    st.write(f'La siguiente gráfica muestra la distribución de supervivencia en función de la edad de los pasajeros. Los datos que has introducido ({sexo} de {edad} años y alojado en {clase} clase), aparecen como una X roja en la gráfica, lo que te ayudará a entender mejor el resultado obtenido.')
    # Empezaremos por la gráfica de distribución de supervivencia por edad
    fig1, ax1 = plt.subplots()
    ax1.scatter(df_titanic['Age'], df_titanic['Survived'], alpha=0.1)
    plt.title("Supervivencia frente a edad")
    plt.xlabel("Edad")
    plt.ylabel("Supervivencia")
    # Añadimos el dato actual como un punto rojo
    plt.plot(X['Age'], Y, 'ro', marker='X')
    # Dibujamos la gráfica
    st.pyplot(fig1)

    # Ahora mostraremos la distribución de supervivencia por Edad y Sexo
    st.write('Además, las siguientes gráficas, muestran la distribución de supervivencia por edad, en función del sexo y la clase de los pasajeros')
    fig2, ax2 = plt.subplots()
    sb.swarmplot(data=df_titanic, x='Sex', y='Age', hue='Survived')
    st.pyplot(fig2)

    # Y finalmente, la distribución de supervivencia por Clase y Sexo
    fig3, ax3 = plt.subplots()
    sb.swarmplot(data=df_titanic, x='Pclass', y='Age', hue='Survived')
    st.pyplot(fig3)


