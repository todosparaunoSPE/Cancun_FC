# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:45:05 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Cancún FC - Análisis Táctico",
    page_icon="⚽",
    layout="wide"
)

# Título principal
st.title("⚽ Cancún FC - Dashboard de Análisis Táctico")
st.markdown("---")

# Información de la vacante
with st.expander("ℹ️ Información de la Vacante - Auxiliar de Análisis Táctico"):
    st.markdown("""
    **Formación requerida:**
    - Licenciatura en entrenamiento deportivo, gestión deportiva, periodismo o comunicación
    - Estadística, matemáticas aplicadas a ciencia de datos
    
    **Certificaciones:**
    - Entrenador y analista táctico
    - Análisis del rendimiento deportivo
    - Big data en el deporte
    
    **Herramientas:** HUDL Sportscode, Nacsport, LongoMatch, Wyacout, InStat, StatsBomb, Excel, Power BI
    
    **Salario base mensual:** $12,000 MXN
    
    **Enviar CV a:** sebastian@cancunfc.com
    """)

# Generar datos de ejemplo para jugadores
np.random.seed(42)
n_players = 50
positions = ['Delantero', 'Mediocampista', 'Defensa', 'Portero']
teams = ['Cancún FC', 'Rival A', 'Rival B', 'Rival C']

data = {
    'Jugador': [f'Jugador {i+1}' for i in range(n_players)],
    'Edad': np.random.randint(18, 35, n_players),
    'Posición': np.random.choice(positions, n_players),
    'Equipo': np.random.choice(teams, n_players),
    'Goles': np.random.randint(0, 20, n_players),
    'Asistencias': np.random.randint(0, 15, n_players),
    'Minutos_Jugados': np.random.randint(500, 2500, n_players),
    'Recuperaciones': np.random.randint(10, 100, n_players),
    'Pases_Completados': np.random.randint(100, 1000, n_players),
    'Precisión_Tiro': np.random.uniform(30, 80, n_players),
    'Eficacia_Defensiva': np.random.uniform(40, 95, n_players)
}

df = pd.DataFrame(data)
df['Rendimiento'] = (df['Goles'] * 0.3 + df['Asistencias'] * 0.25 + 
                     df['Recuperaciones'] * 0.1 + df['Pases_Completados'] * 0.05 + 
                     df['Precisión_Tiro'] * 0.15 + df['Eficacia_Defensiva'] * 0.15)

# Sidebar con filtros
st.sidebar.header("Filtros")
selected_team = st.sidebar.selectbox("Seleccionar Equipo", options=['Todos'] + list(df['Equipo'].unique()))
selected_position = st.sidebar.selectbox("Seleccionar Posición", options=['Todas'] + list(df['Posición'].unique()))

# Aplicar filtros
filtered_df = df.copy()
if selected_team != 'Todos':
    filtered_df = filtered_df[filtered_df['Equipo'] == selected_team]
if selected_position != 'Todas':
    filtered_df = filtered_df[filtered_df['Posición'] == selected_position]

# Mostrar métricas clave
st.header("📊 Métricas Clave del Equipo")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Goles Totales", filtered_df['Goles'].sum())
with col2:
    st.metric("Asistencias Totales", filtered_df['Asistencias'].sum())
with col3:
    st.metric("Rendimiento Promedio", f"{filtered_df['Rendimiento'].mean():.2f}")
with col4:
    st.metric("Jugadores Analizados", filtered_df.shape[0])

# Gráficos de análisis
st.header("📈 Análisis de Rendimiento")

tab1, tab2, tab3, tab4 = st.tabs(["Rendimiento por Posición", "Comparativa de Equipos", "Distribución de Habilidades", "Análisis Individual"])

with tab1:
    fig = px.box(filtered_df, x='Posición', y='Rendimiento', color='Posición',
                 title='Rendimiento por Posición')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    team_stats = df.groupby('Equipo').agg({
        'Goles': 'sum',
        'Asistencias': 'sum',
        'Rendimiento': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=team_stats['Equipo'], y=team_stats['Goles'],
                         name='Goles', marker_color='blue'))
    fig.add_trace(go.Bar(x=team_stats['Equipo'], y=team_stats['Asistencias'],
                         name='Asistencias', marker_color='green'))
    fig.update_layout(title='Comparativa de Equipos - Goles y Asistencias',
                      barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    skills = st.multiselect("Seleccionar Habilidades para Comparar", 
                           options=['Goles', 'Asistencias', 'Recuperaciones', 
                                   'Pases_Completados', 'Precisión_Tiro', 'Eficacia_Defensiva'],
                           default=['Goles', 'Asistencias'])
    
    if skills:
        fig = px.scatter_matrix(filtered_df, dimensions=skills, color='Posición',
                               title='Distribución de Habilidades por Posición')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    selected_player = st.selectbox("Seleccionar Jugador", options=filtered_df['Jugador'].unique())
    player_data = filtered_df[filtered_df['Jugador'] == selected_player].iloc[0]
    
    categories = ['Goles', 'Asistencias', 'Recuperaciones', 
                 'Pases_Completados', 'Precisión_Tiro', 'Eficacia_Defensiva']
    values = [player_data['Goles'], player_data['Asistencias'], 
             player_data['Recuperaciones'], player_data['Pases_Completados'],
             player_data['Precisión_Tiro'], player_data['Eficacia_Defensiva']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=selected_player
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=False,
        title=f"Perfil de Rendimiento - {selected_player}"
    )
    st.plotly_chart(fig, use_container_width=True)

# Modelo de machine learning para predecir rendimiento
st.header("🤖 Modelo Predictivo de Rendimiento")

# Preparar datos para el modelo
X = df[['Edad', 'Goles', 'Asistencias', 'Recuperaciones', 
        'Pases_Completados', 'Precisión_Tiro', 'Eficacia_Defensiva']]
y = pd.cut(df['Rendimiento'], bins=3, labels=['Bajo', 'Medio', 'Alto'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Precisión del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Precisión del modelo de predicción de rendimiento: {accuracy:.2%}")

# Mostrar características importantes
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False)

fig = px.bar(feature_importance, x='Importancia', y='Característica', 
             title='Importancia de Características en la Predicción de Rendimiento')
st.plotly_chart(fig, use_container_width=True)

# Simulador de predicción
st.subheader("Simulador de Potencial de Jugador")
with st.form("simulador_jugador"):
    col1, col2 = st.columns(2)
    with col1:
        edad = st.slider("Edad", 18, 35, 25)
        goles = st.slider("Goles por temporada", 0, 30, 10)
        asistencias = st.slider("Asistencias por temporada", 0, 20, 5)
        recuperaciones = st.slider("Recuperaciones por temporada", 0, 100, 50)
    with col2:
        pases = st.slider("Pases completados por temporada", 0, 1000, 500)
        precision_tiro = st.slider("Precisión de tiro (%)", 30, 80, 60)
        eficacia_def = st.slider("Eficacia defensiva (%)", 40, 95, 75)
    
    submitted = st.form_submit_button("Predecir Rendimiento")
    
    if submitted:
        # Realizar predicción
        input_data = np.array([[edad, goles, asistencias, recuperaciones, pases, precision_tiro, eficacia_def]])
        prediction = model.predict(input_data)[0]
        probability = np.max(model.predict_proba(input_data))
        
        st.success(f"Rendimiento predicho: {prediction} (confianza: {probability:.2%})")
        
        # Mostrar recomendaciones basadas en la predicción
        if prediction == 'Bajo':
            st.info("Recomendación: Enfocarse en mejorar habilidades fundamentales y considerar programa de desarrollo individual.")
        elif prediction == 'Medio':
            st.info("Recomendación: Potencial sólido. Trabajar en aspectos específicos para alcanzar el siguiente nivel.")
        else:
            st.info("Recomendación: Alto potencial. Considerar para posiciones clave y desarrollo de liderazgo.")

# Información adicional
st.markdown("---")
st.markdown("""
### 🎯 Cómo utilizar este dashboard:
1. Utiliza los filtros laterales para analizar equipos o posiciones específicas
2. Explora las diferentes pestañas para diversos tipos de análisis
3. Usa el modelo predictivo para evaluar el potencial de jugadores
4. Exporta los datos para reportes detallados

### 📋 Próximos pasos para el análisis:
- Integración con datos en tiempo real de Wyacout, InStat o StatsBomb
- Análisis de video automatizado con HUDL Sportscode
- Desarrollo de modelos predictivos más avanzados
- Creación de reportes automatizados para el cuerpo técnico
""")

# Pie de página
st.markdown("---")
st.markdown("© 2023 Cancún FC - Departamento de Análisis Táctico | [Enviar CV](mailto:sebastian@cancunfc.com)")