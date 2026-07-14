import sys
import types
import os
import pandas as pd
import joblib
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import sklearn.compose._column_transformer
import plotly.graph_objects as go
import plotly.express as px

# ==================== PARCHES DE COMPATIBILIDAD ====================
class DummyRemainderColsList(list):
    pass

sklearn.compose._column_transformer._RemainderColsList = DummyRemainderColsList
sys.modules['sklearn.compose._column_transformer']._RemainderColsList = DummyRemainderColsList

@property
def fill_dtype_patch(self):
    return getattr(self, '_fit_dtype', None)

SimpleImputer._fill_dtype = fill_dtype_patch

_original_get_params = xgb.XGBModel.get_params

def safe_get_params(self, deep=True):
    try:
        return _original_get_params(self, deep=deep)
    except AttributeError as e:
        missing_attr = str(e).split("'")[-2]
        setattr(self, missing_attr, None)
        return safe_get_params(self, deep=deep)

xgb.XGBModel.get_params = safe_get_params
# ================================================================

# ==================== CONFIGURACIÓN INICIAL ====================
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño compacto
st.markdown("""
<style>
    /* Estilo para el botón más compacto */
    .stButton button {
        width: auto !important;
        min-width: 200px !important;
        max-width: 300px !important;
        margin: 0 auto !important;
        padding: 0.5rem 2rem !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
    }
    
    /* Contenedor centrado para el botón */
    .button-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    /* Estilo para el gauge más compacto */
    .gauge-container {
        max-width: 400px !important;
        margin: 0 auto !important;
    }
    
    /* Estilo para tarjetas de categoría */
    .category-card {
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
        border-left: 5px solid;
        background: #f8f9fa;
    }
    
    .category-card h3 {
        margin: 0;
        font-size: 1.1rem;
    }
    
    .category-card p {
        margin: 5px 0 0 0;
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Estilo para métricas compactas */
    .compact-metric {
        background: white;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    
    /* Leyenda compacta */
    .legend-item {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCIONES AUXILIARES ====================
@st.cache_data
def cargar_modelo(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def obtener_categoria_crediticia(prob_incumplimiento):
    """Determina la categoría crediticia basada en la probabilidad de incumplimiento"""
    if prob_incumplimiento < 0.2:
        return "Normal", "✅", "#2ecc71", "Bajo riesgo crediticio"
    elif prob_incumplimiento < 0.35:
        return "Problemas potenciales", "⚠️", "#f39c12", "Riesgo moderado - requiere seguimiento"
    elif prob_incumplimiento < 0.5:
        return "Deficiente", "🔶", "#e67e22", "Alto riesgo - supervisión estricta"
    elif prob_incumplimiento < 0.65:
        return "Dudoso", "🔴", "#e74c3c", "Muy alto riesgo - posible incumplimiento"
    else:
        return "Pérdida", "💀", "#c0392b", "Riesgo extremo - incumplimiento probable"

def crear_grafico_categoria_crediticia(prob_incumplimiento):
    """Crea un gráfico de gauge compacto"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_incumplimiento * 100,
        title={'text': "Nivel de Riesgo", 'font': {'size': 14}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 10}},
            'bar': {'color': "#2c3e50", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, 20], 'color': '#2ecc71'},
                {'range': [20, 35], 'color': '#f39c12'},
                {'range': [35, 50], 'color': '#e67e22'},
                {'range': [50, 65], 'color': '#e74c3c'},
                {'range': [65, 100], 'color': '#c0392b'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.6,
                'value': prob_incumplimiento * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        width=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'size': 12, 'family': 'Arial'}
    )
    
    return fig

def mostrar_tarjeta_categoria(categoria, icono, color, descripcion):
    """Muestra una tarjeta visual con la categoría crediticia"""
    st.markdown(f"""
    <div class="category-card" style="border-left-color: {color};">
        <h3 style="color: {color};">
            {icono} {categoria}
        </h3>
        <p>{descripcion}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== CARGA DEL MODELO ====================
MODEL_PATH = "credit_scoring_xgboost.joblib"
artefacto = cargar_modelo(MODEL_PATH)

if artefacto is None:
    st.error(f"❌ No se encontró el archivo `{MODEL_PATH}` en el directorio actual.")
    st.stop()

# ==================== EXTRACCIÓN DE METADATOS ====================
pipeline = artefacto["pipeline"]
esquema_entrada = artefacto["esquema_entrada"]
ejemplo_entrada = artefacto["ejemplo_entrada"]
umbral_decision = artefacto.get("umbral_decision", 0.5)

# ==================== INTERFAZ PRINCIPAL ====================
st.title("💳 Sistema de Evaluación de Crédito")
st.markdown("Evaluación de riesgo crediticio en tiempo real")

# ==================== INPUTS DEL USUARIO ====================
st.subheader("👤 Datos del Solicitante")
st.caption("Modifica los valores para evaluar el riesgo en tiempo real.")

# Organización con pestañas
tab1, tab2, tab3 = st.tabs(["💰 Financiero", "📊 Historial", "📋 Demográfico"])

datos_usuario = {}

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        datos_usuario['ingreso_mensual'] = st.number_input(
            "Ingreso Mensual (S/.)", 
            min_value=0.0, 
            value=float(ejemplo_entrada['ingreso_mensual']),
            step=100.0,
            key="ingreso"
        )
        datos_usuario['ratio_deuda_ingreso'] = st.slider(
            "Ratio Deuda/Ingreso", 
            min_value=0.0, 
            max_value=2.0, 
            value=float(ejemplo_entrada['ratio_deuda_ingreso']),
            step=0.01,
            key="deuda"
        )
    with col2:
        datos_usuario['utilizacion_credito'] = st.slider(
            "Utilización Crédito (0-1)", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ejemplo_entrada['utilizacion_credito']),
            step=0.01,
            key="utilizacion"
        )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        datos_usuario['antiguedad_laboral'] = st.number_input(
            "Antigüedad Laboral (Años)", 
            min_value=0.0, 
            value=float(ejemplo_entrada['antiguedad_laboral']),
            step=0.5,
            key="laboral"
        )
        datos_usuario['antiguedad_crediticia'] = st.number_input(
            "Antigüedad Crediticia (Años)", 
            min_value=0.0, 
            value=float(ejemplo_entrada['antiguedad_crediticia']),
            step=0.5,
            key="crediticia"
        )
        datos_usuario['numero_productos_crediticios'] = st.number_input(
            "Número de Productos Crediticios", 
            min_value=0, 
            value=int(ejemplo_entrada['numero_productos_crediticios']),
            step=1,
            key="productos"
        )
    with col2:
        datos_usuario['consultas_ultimos_6_meses'] = st.number_input(
            "Consultas Últimos 6 Meses", 
            min_value=0, 
            value=int(ejemplo_entrada['consultas_ultimos_6_meses']),
            step=1,
            key="consultas"
        )
        datos_usuario['atrasos_ultimos_12_meses'] = st.number_input(
            "Atrasos Últimos 12 Meses", 
            min_value=0, 
            value=int(ejemplo_entrada['atrasos_ultimos_12_meses']),
            step=1,
            key="atrasos"
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        datos_usuario['monto_solicitado'] = st.number_input(
            "Monto Solicitado (S/.)", 
            min_value=0.0, 
            value=float(ejemplo_entrada['monto_solicitado']),
            step=500.0,
            key="monto"
        )
        datos_usuario['plazo_meses'] = st.number_input(
            "Plazo (Meses)", 
            min_value=1, 
            value=int(ejemplo_entrada['plazo_meses']),
            step=1,
            key="plazo"
        )
        datos_usuario['ratio_cuota_ingreso'] = st.slider(
            "Ratio Cuota/Ingreso", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ejemplo_entrada['ratio_cuota_ingreso']),
            step=0.01,
            key="cuota"
        )
    with col2:
        # Variables categóricas
        campos_categoricos = ['tipo_empleo', 'tipo_vivienda', 'destino_credito']
        for campo in campos_categoricos:
            options = esquema_entrada[campo]['categorias_conocidas']
            default_index = options.index(ejemplo_entrada[campo])
            datos_usuario[campo] = st.selectbox(
                campo.replace('_', ' ').title(),
                options=options,
                index=default_index,
                key=f"select_{campo}"
            )

# ==================== BOTÓN CENTRADO ====================
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
calcular = st.button("📊 Calcular Scoring de Riesgo", type="primary", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ==================== EVALUACIÓN ====================
if calcular:
    try:
        df_cliente = pd.DataFrame([datos_usuario])
        
        with st.spinner("Calculando riesgo crediticio..."):
            probabilidades = pipeline.predict_proba(df_cliente)
            prob_incumplimiento = probabilidades[0][1]
        
        # Obtener categoría crediticia
        categoria, icono, color, descripcion = obtener_categoria_crediticia(prob_incumplimiento)
        
        # ==================== VISUALIZACIÓN DE CATEGORÍA CREDITICIA ====================
        st.markdown("---")
        st.subheader("📊 Calificación Crediticia")
        
        # Layout de 3 columnas para mejor distribución
        col_gaug, col_info, col_metrics = st.columns([2, 2, 1])
        
        with col_gaug:
            # Gráfico de gauge compacto
            fig_gauge = crear_grafico_categoria_crediticia(prob_incumplimiento)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        with col_info:
            st.markdown("### Resultado")
            mostrar_tarjeta_categoria(categoria, icono, color, descripcion)
            
            # Probabilidad con formato compacto
            st.metric(
                label="Riesgo",
                value=f"{prob_incumplimiento:.1%}",
                delta=f"Umbral: {umbral_decision:.0%}",
                delta_color="inverse"
            )
        
        with col_metrics:
            st.markdown("### Categorías")
            # Leyenda compacta
            categorias_legend = [
                ("Normal", "#2ecc71"),
                ("Problemas", "#f39c12"),
                ("Deficiente", "#e67e22"),
                ("Dudoso", "#e74c3c"),
                ("Pérdida", "#c0392b")
            ]
            for cat, col in categorias_legend:
                st.markdown(f"""
                <span class="legend-item" style="background:{col}20; border-color:{col}; color:{col};">
                    ● {cat}
                </span>
                """, unsafe_allow_html=True)
        
        # ==================== CLASIFICACIÓN FINAL ====================
        st.markdown("---")
        st.subheader("📋 Decisión Final")
        
        # Clasificación según umbral de decisión
        if prob_incumplimiento >= umbral_decision:
            estado = "❌ RECHAZADO"
            color_estado = "error"
            mensaje = "El solicitante presenta alto riesgo de incumplimiento"
        elif prob_incumplimiento >= (umbral_decision * 0.6):
            estado = "⚠️ REVISIÓN MANUAL"
            color_estado = "warning"
            mensaje = "El solicitante requiere análisis adicional"
        else:
            estado = "✅ APROBADO"
            color_estado = "success"
            mensaje = "El solicitante presenta bajo riesgo de incumplimiento"
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if color_estado == "success":
                st.success(f"### {estado}\n{mensaje}")
            elif color_estado == "warning":
                st.warning(f"### {estado}\n{mensaje}")
            else:
                st.error(f"### {estado}\n{mensaje}")
        
        with col2:
            st.info(f"""
            **Categoría Crediticia:** {icono} **{categoria}**
            
            **Probabilidad:** {prob_incumplimiento:.1%}
            
            **Recomendación:** {descripcion}
            """)
        
        # ==================== DETALLES EXPANDIBLES ====================
        with st.expander("📊 Ver análisis detallado", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Factores clave:**")
                st.write(f"- Ratio Deuda/Ingreso: {datos_usuario['ratio_deuda_ingreso']:.2f}")
                st.write(f"- Utilización Crédito: {datos_usuario['utilizacion_credito']:.1%}")
                st.write(f"- Atrasos: {datos_usuario['atrasos_ultimos_12_meses']} en 12 meses")
            with col2:
                st.write("**Perfil del cliente:**")
                st.write(f"- Antigüedad Crediticia: {datos_usuario['antiguedad_crediticia']} años")
                st.write(f"- Ingreso Mensual: S/. {datos_usuario['ingreso_mensual']:,.2f}")
                st.write(f"- Monto Solicitado: S/. {datos_usuario['monto_solicitado']:,.2f}")
        
    except Exception as e:
        st.error(f"Error durante la evaluación: {str(e)}")
        st.info("Verifica que todos los campos estén completos correctamente.")

# ==================== FOOTER ====================
#st.markdown("---")
#st.caption("💡 Modelo XGBoost optimizado con umbral de decisión ajustable")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 Información del Modelo")
    #st.metric("Nombre", artefacto.get('nombre', 'XGBoost Credit Scoring'))
    #st.metric("Versión", artefacto.get('version', '1.0'))
    st.metric("Umbral Decisión", f"{umbral_decision:.1%}")
    
    st.markdown("---")
    st.markdown("### 🏷️ Categorías")
    categorias_sidebar = [
        ("✅ Normal", "#2ecc71"),
        ("⚠️ Problemas potenciales", "#f39c12"),
        ("🔶 Deficiente", "#e67e22"),
        ("🔴 Dudoso", "#e74c3c"),
        ("💀 Pérdida", "#c0392b")
    ]
    for cat, col in categorias_sidebar:
        st.markdown(f'<span style="color: {col}; font-size: 14px;">●</span> {cat}', unsafe_allow_html=True)
