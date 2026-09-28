from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Detector de Objetos ☕",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────  ESTILOS (TEMA CAFÉ)  ─────────────────────────
st.markdown("""
<style>
:root {
    color-scheme: light;
    --crema:      #F7EFE5;
    --latte:      #EADBC8;
    --papel:      #FFFBF6;
    --moka:       #C9A27E;
    --caramelo:   #A9713F;
    --cafe:       #8B5E3C;
    --chocolate:  #3B2622;
    --espresso:   #2E1F1B;
}

/* ── Fondo general y texto ── */
.stApp {
    background-color: var(--crema) !important;
    color: var(--espresso) !important;
}
header[data-testid="stHeader"] { background-color: var(--crema) !important; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--espresso) !important;
}
.stApp p, .stApp li, .stApp label, .stApp span,
.stApp div[data-testid="stMarkdownContainer"],
.stApp div[data-testid="stCaptionContainer"] {
    color: var(--espresso);
}

/* ── Barra lateral ── */
section[data-testid="stSidebar"] {
    background-color: var(--latte) !important;
    border-right: 2px solid var(--moka);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {
    color: var(--espresso) !important;
}

/* ── Banner principal (texto claro sobre fondo oscuro) ── */
.stApp .hero {
    background: linear-gradient(135deg, #3B2622, #7A5033);
    padding: 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 18px rgba(46, 31, 27, 0.30);
}
.stApp .hero h1 { color: #FFF8F0 !important; margin: 0 0 .4rem 0; font-size: 2.2rem; }
.stApp .hero p  { color: #F5E9DA !important; margin: 0; font-size: 1.05rem; }

/* ── Tarjetas (fondo claro, texto oscuro) ── */
.stApp .card {
    background-color: var(--papel);
    border: 1px solid var(--moka);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(46, 31, 27, 0.08);
}
.stApp .card, .stApp .card * { color: var(--espresso) !important; }
.stApp .card h3 { margin-top: 0; }

/* ── Cuadro explicativo de la barra lateral ── */
section[data-testid="stSidebar"] .about-box {
    background-color: var(--papel);
    border-left: 5px solid var(--caramelo);
    border-radius: 10px;
    padding: .9rem 1rem;
    margin-bottom: 1rem;
    font-size: .93rem;
    line-height: 1.5;
}
section[data-testid="stSidebar"] .about-box,
section[data-testid="stSidebar"] .about-box * {
    color: var(--espresso) !important;
}

/* ── Métricas ── */
div[data-testid="stMetric"] {
    background-color: var(--papel);
    border: 1px solid var(--moka);
    border-radius: 14px;
    padding: .9rem 1rem;
    box-shadow: 0 2px 8px rgba(46, 31, 27, 0.08);
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color: var(--chocolate) !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color: var(--cafe) !important; }

/* ── Cámara ── */
div[data-testid="stCameraInput"] {
    background-color: var(--latte);
    border: 1px solid var(--moka);
    border-radius: 14px;
    padding: .6rem;
}
div[data-testid="stCameraInput"] label,
div[data-testid="stCameraInput"] p { color: var(--espresso) !important; }

/* ── Botones (texto claro sobre café) ── */
.stButton > button,
div[data-testid="stCameraInput"] button {
    background-color: var(--cafe) !important;
    border: none !important;
    border-radius: 10px !important;
}
.stButton > button *,
div[data-testid="stCameraInput"] button * { color: #FFF8F0 !important; }
.stButton > button:hover,
div[data-testid="stCameraInput"] button:hover { background-color: var(--caramelo) !important; }

/* ── Alertas (st.info, st.error, etc.) ── */
div[data-testid="stAlert"] {
    background-color: var(--papel) !important;
    border: 1px solid var(--moka);
    border-radius: 12px;
}
div[data-testid="stAlert"], div[data-testid="stAlert"] * { color: var(--espresso) !important; }

/* ── Pestañas ── */
button[data-baseweb="tab"], button[data-baseweb="tab"] * { color: var(--chocolate) !important; }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] * {
    color: var(--cafe) !important; font-weight: 600;
}
div[data-baseweb="tab-highlight"] { background-color: var(--cafe) !important; }

/* ── Inputs numéricos ── */
div[data-baseweb="input"], div[data-baseweb="input"] input, div[data-baseweb="base-input"] {
    background-color: var(--papel) !important;
    color: var(--espresso) !important;
}
div[data-baseweb="input"] button { background-color: var(--latte) !important; }
div[data-baseweb="input"] button * { color: var(--espresso) !important; }

/* ── Sliders ── */
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: var(--espresso) !important; }

/* ── Imágenes y tablas ── */
div[data-testid="stImage"] img { border-radius: 12px; border: 2px solid var(--moka); }
div[data-testid="stDataFrame"] {
    border: 1px solid var(--moka);
    border-radius: 10px;
    background-color: var(--papel);
}

/* ── Separadores ── */
hr { border-color: var(--moka) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────  MODELO  ─────────────────────────
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Ups, no pude cargar el modelo: {str(e)}")
        return None


# ─────────────────────────  ENCABEZADO  ─────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Detector de Objetos</h1>
    <p>Toma una foto con tu cámara y descubre qué objetos aparecen en ella. ¡Es rápido y sencillo!</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("☕ Preparando el modelo YOLOv5, un momento por favor..."):
    model = load_model()

if model:
    # ─────────────────────────  BARRA LATERAL  ─────────────────────────
    with st.sidebar:
        st.markdown("## ☕ ¿Qué es este sitio?")
        st.markdown("""
        <div class="about-box">
            Esta aplicación usa <b>inteligencia artificial (YOLOv5)</b> para reconocer
            objetos en las fotos que tomas con tu cámara: personas, celulares, sillas,
            tazas y muchas cosas más.<br><br>
            <b>¿Para qué sirve?</b> Para explorar cómo "ve" una IA el mundo, hacer
            pruebas de visión por computador o simplemente divertirte identificando
            lo que hay a tu alrededor. 📸
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## ⚙️ Parámetros")
        st.caption("Ajusta estos valores para afinar la detección a tu gusto.")

        conf_threshold = st.slider(
            "Confianza mínima", 0.0, 1.0, 0.25, 0.01,
            help="Qué tan seguro debe estar el modelo para mostrar un objeto. Más alto = menos detecciones, pero más precisas."
        )
        iou_threshold = st.slider(
            "Umbral IoU", 0.0, 1.0, 0.45, 0.01,
            help="Controla cuánto pueden superponerse los recuadros antes de considerarse el mismo objeto."
        )
        max_det = st.number_input(
            "Detecciones máximas", 10, 2000, 1000, 10,
            help="Cantidad máxima de objetos que se mostrarán en la imagen."
        )

    # ─────────────────────────  PASO 1: CAPTURA  ─────────────────────────
    st.markdown("### 📷 Paso 1: Toma tu foto")
    cam_col, tips_col = st.columns([2, 1], gap="large")

    with cam_col:
        picture = st.camera_input("Capturar imagen", key="camera", label_visibility="collapsed")

    with tips_col:
        st.markdown("""
        <div class="card">
            <h3>💡 Consejos</h3>
            <ul>
                <li>Busca un lugar con buena luz.</li>
                <li>Acércate un poco a los objetos.</li>
                <li>Si no detecta nada, baja la <b>confianza mínima</b> en la barra lateral.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────  PASO 2: RESULTADOS  ─────────────────────────
    if picture:
        bytes_data = picture.getvalue()

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img = np.array(pil_img)[..., ::-1]  # RGB → BGR para que YOLO procese bien

        with st.spinner("🔎 Analizando tu imagen..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Algo salió mal durante la detección: {str(e)}")
                st.stop()

        result = results[0]
        boxes = result.boxes
        annotated = result.plot()               # devuelve BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]   # BGR → RGB sin cv2

        st.markdown("---")
        st.markdown("### 🎯 Paso 2: Mira lo que encontré")

        has_detections = boxes is not None and len(boxes) > 0

        if has_detections:
            label_names = model.names
            category_count = {}
            category_conf = {}

            for box in boxes:
                cat = int(box.cls.item())
                conf = float(box.conf.item())
                category_count[cat] = category_count.get(cat, 0) + 1
                category_conf.setdefault(cat, []).append(conf)

            data = [
                {
                    "Categoría": label_names[cat],
                    "Cantidad": count,
                    "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                }
                for cat, count in category_count.items()
            ]
            df = pd.DataFrame(data)

            total_objs = len(boxes)
            total_cats = len(category_count)
            all_confs = [c for lst in category_conf.values() for c in lst]
            avg_conf = float(np.mean(all_confs))

            m1, m2, m3 = st.columns(3)
            m1.metric("🧩 Objetos detectados", total_objs)
            m2.metric("🗂️ Categorías distintas", total_cats)
            m3.metric("✅ Confianza promedio", f"{avg_conf:.0%}")
            st.write("")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 🖼️ Imagen con detecciones")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.markdown("#### 📋 Resumen de objetos")
            if has_detections:
                tab_tabla, tab_grafico = st.tabs(["📄 Tabla", "📊 Gráfico"])
                with tab_tabla:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                with tab_grafico:
                    st.bar_chart(df.set_index("Categoría")["Cantidad"], color="#8B5E3C")
            else:
                st.info("No encontré objetos con los parámetros actuales. 🤔")
                st.caption("Prueba bajando la confianza mínima en la barra lateral.")
    else:
        st.info("👆 Toma una foto para comenzar. ¡Los resultados aparecerán aquí!")

else:
    st.error("No se pudo cargar el modelo. Revisa las dependencias e inténtalo de nuevo.")
    st.stop()

st.markdown("---")
st.caption("☕ **Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
