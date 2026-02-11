import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Configuración
# ----------------------------
st.set_page_config(
    page_title="Titanic Dataset (Built-in)",
    layout="wide"
)

st.title("🚢 Dataset Titanic (base integrada de Python)")
st.write(
    "Visualización interactiva usando `seaborn.load_dataset()` "
    "— sin archivos CSV."
)

# ----------------------------
# Cargar datos
# ----------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

# ----------------------------
# Vista general
# ----------------------------
st.subheader("📋 Vista previa")
st.dataframe(df.head(20), use_container_width=True)

# ----------------------------
# Métricas rápidas
# ----------------------------
st.subheader("📊 Métricas clave")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pasajeros", len(df))

with col2:
    st.metric(
        "Supervivencia (%)",
        f"{df['survived'].mean() * 100:.1f}%"
    )

with col3:
    st.metric(
        "Edad media",
        f"{df['age'].mean():.1f}"
    )

# ----------------------------
# Filtros
# ----------------------------
st.subheader("🔍 Filtros")

col1, col2, col3 = st.columns(3)

with col1:
    sex_filter = st.multiselect(
        "Sexo",
        df["sex"].dropna().unique(),
        default=df["sex"].dropna().unique()
    )

with col2:
    class_filter = st.multiselect(
        "Clase",
        df["class"].dropna().unique(),
        default=df["class"].dropna().unique()
    )

with col3:
    alone_filter = st.selectbox(
        "Viajaba solo",
        ["Todos", "Sí", "No"]
    )

filtered_df = df[
    (df["sex"].isin(sex_filter)) &
    (df["class"].isin(class_filter))
]

if alone_filter == "Sí":
    filtered_df = filtered_df[filtered_df["alone"] == True]
elif alone_filter == "No":
    filtered_df = filtered_df[filtered_df["alone"] == False]

st.write(f"Registros filtrados: **{len(filtered_df)}**")
st.dataframe(filtered_df, use_container_width=True)

# ----------------------------
# Gráficos
# ----------------------------
st.subheader("📈 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Supervivencia por sexo**")
    fig, ax = plt.subplots()
    sns.barplot(
        data=filtered_df,
        x="sex",
        y="survived",
        ax=ax
    )
    ax.set_ylabel("Probabilidad de supervivencia")
    st.pyplot(fig)

with col2:
    st.markdown("**Distribución de edades**")
    fig, ax = plt.subplots()
    filtered_df["age"].dropna().plot(
        kind="hist",
        bins=30,
        ax=ax
    )
    ax.set_xlabel("Edad")
    st.pyplot(fig)

# ----------------------------
# Estadísticas detalladas
# ----------------------------
with st.expander("📊 Ver estadísticas completas"):
    st.write(filtered_df.describe(include="all"))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Dataset: seaborn.load_dataset('titanic') | Streamlit")
