import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_clean_data():
    data= pd.read_csv("data/data.csv")#data\data.csv
    
    #Eliminamos columnas innecesarias
    data = data.drop(['Unnamed: 32','id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data


def add_sidebar():
    st.sidebar.header("Mediciones de Nucleos Celulares")
    data = get_clean_data()

    # Definir las etiquetas
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}



    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
        
    return input_dict



def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict



def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    
    categories = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness',
                  'Concavity','Concave Points','Symetry','Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],input_data['texture_mean'],input_data['perimeter_mean'],
            input_data['area_mean'],input_data['smoothness_mean'],input_data['compactness_mean'],
            input_data['concavity_mean'],input_data['concave points_mean'], input_data['symmetry_mean'], 
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Valor Promedio'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],input_data['texture_se'],input_data['perimeter_se'],
            input_data['area_se'],input_data['smoothness_se'],input_data['compactness_se'],
            input_data['concavity_se'],input_data['concave points_se'], input_data['symmetry_se'], 
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Error Estandar'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],input_data['texture_worst'],input_data['perimeter_worst'],
            input_data['area_worst'],input_data['smoothness_worst'],input_data['compactness_worst'],
            input_data['concavity_worst'],input_data['concave points_worst'], input_data['symmetry_worst'], 
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Peor Valor'
    ))


    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    #fig.show()
    return fig



def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl','rb'))
    scaler = pickle.load(open('model/scaler.pkl','rb'))

    #input_array = np.array(list(input_data.values()))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Escalamos los valores que vienen del slider
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Predicción de grupos de células")
    st.write("El grupo celular es:")


    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benigno</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Maligno</span>", unsafe_allow_html=True)
    

    st.write("Probabilidad de ser Benigno: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probabilidad de ser Maligno: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("Esta aplicación puede ayudar a los profesionales médicos a realizar un diagnóstico, pero no debe utilizarse como sustituto de un diagnóstico profesional")
    
    #st.write(prediction)





def main():
    st.set_page_config(
        page_title="Predictor de Cancer de mama",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Barra lateral
    input_data = add_sidebar()

    with st.container():
        st.title("PREDICTOR DE CANCER DE MAMA")
        st.write("Conecte esta aplicación a su laboratorio de citología para ayudar a diagnosticar el cáncer de mama a partir de su muestra de tejido. Esta aplicación predice utilizando un modelo de aprendizaje automático si una masa mamaria es benigna o maligna en función de las mediciones que recibe de su laboratorio de citosis. También puede actualizar las medidas a mano usando los controles deslizantes en la barra lateral.")

    col1, col2 = st.columns([4,1])
    with col1:
        #st.write("Columna 1")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        #st.write("Columna 2")
        add_predictions(input_data)
if __name__ == "__main__":
    main()