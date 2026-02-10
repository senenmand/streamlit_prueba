import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Configuración inicial
# ----------------------------
st.set_page_config(
    page_title="Titanic Dataset - Kaggle",
    layout="wide"
)

st.title("🚢 Visualización del Dataset Titanic (Kaggle)")
st.write(
    "Aplicación interactiva para explorar el famoso dataset del Titanic."
)

# ----------------------------
# Carga de datos
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# ----------------------------
# Vista previa
# ----------------------------
st.subheader("📋 Vista previa de los datos")
st.dataframe(df.head(20), use_container_width=True)

# ----------------------------
# Estadísticas generales
# ----------------------------
st.subheader("📊 Estadísticas generales")
st.write(df.describe())

# ----------------------------
# Filtros
# ----------------------------
st.subheader("🔍 Filtros interactivos")

col1, col2, col3 = st.columns(3)

with col1:
    sex_filter = st.multiselect(
        "Sexo",
        options=df["Sex"].dropna().unique(),
        default=df["Sex"].dropna().unique()
    )

with col2:
    class_filter = st.multiselect(
        "Clase",
        options=sorted(df["Pclass"].unique()),
        default=sorted(df["Pclass"].unique())
    )

with col3:
    survived_filter = st.selectbox(
        "Supervivencia",
        options=["Todos", "Sobrevivió", "No sobrevivió"]
    )

filtered_df = df[
    (df["Sex"].isin(sex_filter)) &
    (df["Pclass"].isin(class_filter))
]

if survived_filter == "Sobrevivió":
    filtered_df = filtered_df[filtered_df["Survived"] == 1]
elif survived_filter == "No sobrevivió":
    filtered_df = filtered_df[filtered_df["Survived"] == 0]

st.write(f"Registros filtrados: **{len(filtered_df)}**")
st.dataframe(filtered_df, use_container_width=True)

# ----------------------------
# Visualizaciones
# ----------------------------
st.subheader("📈 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Supervivencia por sexo**")
    surv_sex = filtered_df.groupby("Sex")["Survived"].mean()
    fig, ax = plt.subplots()
    surv_sex.plot(kind="bar", ax=ax)
    ax.set_ylabel("Probabilidad de supervivencia")
    st.pyplot(fig)

with col2:
    st.markdown("**Distribución de edades**")
    fig, ax = plt.subplots()
    filtered_df["Age"].dropna().plot(kind="hist", bins=30, ax=ax)
    ax.set_xlabel("Edad")
    st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Dataset: Titanic - Kaggle | App creada con Streamlit")
