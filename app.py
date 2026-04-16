import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder

# ===== KONFIGURACJA =====
MODEL_NAME = "welcome_survey_clustering_pipeline_v2.pkl"
DATA = "welcome_survey_extended_1.csv"
CLUSTER_NAMES = "welcome_survey_cluster_names_and_descriptions_v1.json"

st.set_page_config(
    page_title="🌟 Znajdź Swoją Grupę!",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== GLAMOUR CSS =====
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    .cluster-card {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        margin: 20px 0;
        text-align: center;
    }
    
    .cluster-card h1 {
        font-size: 3em;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .emoji-art {
        font-size: 5em;
        margin: 20px 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(236, 72, 153, 0.3);
        margin: 10px 0;
    }
    
    .metric-box h2 {
        color: #EC4899;
        font-size: 2.5em;
        margin: 10px 0;
    }
    
    .metric-box p {
        color: #C084FC;
        font-size: 1.2em;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0a1e 0%, #1a0a2e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        background-color: #2d1b4e;
        color: white;
        font-size: 1.1em;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
    }
    
    h1, h2, h3, h4, p, span, div, label {
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===== FUNKCJE POMOCNICZE =====


@st.cache_resource
def get_model():
    return joblib.load(MODEL_NAME)


@st.cache_data
def get_cluster_info():
    with open(CLUSTER_NAMES, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_all_data():
    df = pd.read_csv(DATA, sep=";")
    model = get_model()
    df["Cluster"] = model.predict(df)
    return df


def get_color_palette():
    return ["#8B5CF6", "#A78BFA", "#C084FC", "#E879F9", "#EC4899", "#F472B6"]


def create_emoji_art(icon):
    """Tworzy większą kompozycję emoji"""
    emoji_list = list(icon)
    if len(emoji_list) >= 2:
        # Duża kompozycja 3x3
        art = f"""
        {emoji_list[0]} {emoji_list[1]} {emoji_list[0]}
        {emoji_list[1]} {emoji_list[0]} {emoji_list[1]}
        {emoji_list[0]} {emoji_list[1]} {emoji_list[0]}
        """
    else:
        # Pojedyncze emoji w układzie
        art = f"""
        {icon} {icon} {icon}
        {icon} ✨ {icon}
        {icon} {icon} {icon}
        """
    return art


# ===== WCZYTAJ DANE =====
model = get_model()
all_data = get_all_data()
cluster_info = get_cluster_info()

# ===== SIDEBAR - WYBÓR =====
st.sidebar.markdown("## ✨ Powiedz nam o sobie")
st.sidebar.markdown("---")

age_options = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", "unknown"]
age = st.sidebar.selectbox("🎂 Wiek", age_options)

edu_options = ["Podstawowe", "Średnie", "Wyższe"]
edu_level = st.sidebar.selectbox("🎓 Wykształcenie", edu_options)

animal_options = [
    "Brak ulubionych",
    "Psy",
    "Koty",
    "Inne",
    "Koty i Psy",
    "Świnki morskie",
    "Żółwie",
    "Konie",
    "Rybki",
    "Papugi",
    "Króliki",
    "Chomiki",
]
fav_animals = st.sidebar.selectbox("🐾 Ulubione zwierzęta", animal_options)

place_options = [
    "Nad wodą",
    "W lesie",
    "W górach",
    "Inne",
    "Nad morzem",
    "W parkach",
    "Plaża",
    "W mieście",
    "Pustynia",
    "Na wsi",
    "Jaskinie",
]
fav_place = st.sidebar.selectbox("📍 Ulubione miejsce", place_options)

gender_options = ["Kobieta", "Mężczyzna", "Agent AI"]
gender = st.sidebar.radio("⚧️ Płeć", gender_options)

st.sidebar.markdown("---")
find_button = st.sidebar.button("🔍 ZNAJDŹ MOJĄ GRUPĘ!", use_container_width=True)

# Sprawdź czy dane się zmieniły
user_data_key = f"{age}_{edu_level}_{fav_animals}_{fav_place}_{gender}"

# Inicjalizuj session state
if "last_search" not in st.session_state:
    st.session_state["last_search"] = None

# Automatycznie aktualizuj gdy użytkownik zmienia wartości LUB klika przycisk
if find_button:
    st.session_state["last_search"] = user_data_key
    st.session_state["cluster_found"] = True
    st.rerun()  # Wymuś odświeżenie
elif (
    st.session_state["last_search"] != user_data_key
    and st.session_state["last_search"] is not None
):
    st.session_state["last_search"] = user_data_key
    st.session_state["cluster_found"] = True
    st.rerun()  # Wymuś odświeżenie

if st.session_state.get("cluster_found", False) and st.session_state["last_search"]:

    # Przygotuj dane użytkownika
    person_df = pd.DataFrame(
        [
            {
                "age": age,
                "edu_level": edu_level,
                "fav_animals": fav_animals,
                "fav_place": fav_place,
                "gender": gender,
            }
        ]
    )

    # Predykcja klastra
    cluster_id = model.predict(person_df)[0]
    cluster_key = f"Cluster {cluster_id}"
    cluster_data = cluster_info[cluster_key]

    # Filtruj dane dla tego klastra
    cluster_df = all_data[all_data["Cluster"] == cluster_id]

    # ===== NAGŁÓWEK KLASTRA =====
    st.markdown(
        f"""
    <div class="cluster-card">
        <div class="emoji-art">{create_emoji_art(cluster_data['icon'])}</div>
        <h1>{cluster_data['name']}</h1>
        <p style='font-size: 1.3em; margin: 20px 0;'>{cluster_data['description']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metryki
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-box">
            <h2>👥 {len(cluster_df)}</h2>
            <p>Osób w grupie</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        most_common_age = cluster_df["age"].mode()[0] if len(cluster_df) > 0 else "N/A"
        st.markdown(
            f"""
        <div class="metric-box">
            <h2>🎂 {most_common_age}</h2>
            <p>Najczęstszy wiek</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        # Dynamiczne wyświetlanie wybranej płci
        gender_emoji = {"Kobieta": "👩", "Mężczyzna": "👨", "Agent AI": "🤖"}
        selected_gender_pct = (
            (cluster_df["gender"] == gender).sum() / len(cluster_df) * 100
            if len(cluster_df) > 0
            else 0
        )

        st.markdown(
            f"""
        <div class="metric-box">
            <h2>{gender_emoji.get(gender, '👤')} {selected_gender_pct:.0f}%</h2>
            <p>{gender} w grupie</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ===== ZAKŁADKI =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "👤 Twój Klaster",
            "🎂 Wiek",
            "🎓 Wykształcenie",
            "🐾 Lubimy",
            "📍 Czujemy się najlepiej",
        ]
    )

    # TAB 1: Twój Klaster
    with tab1:
        st.markdown("### 🎨 Szczegółowa Analiza Twojej Grupy")

        col1, col2 = st.columns(2)

        with col1:
            # Rozkład wszystkich klastrów
            all_clusters = all_data["Cluster"].value_counts().sort_index()

            # Przygotuj DataFrame z nazwami
            chart_data = []
            for cluster_num in all_clusters.index:
                cluster_key = f"Cluster {cluster_num}"
                cluster_name = cluster_info[cluster_num]["name"]
                cluster_emoji = cluster_info[cluster_num]["icon"]

                # Krótka nazwa (pierwsze 2-3 słowa)
                short_name = " ".join(cluster_name.split()[:3])
                if len(short_name) > 20:
                    short_name = short_name[:17] + "..."

                chart_data.append(
                    {
                        "short_name": f"{cluster_emoji} {short_name}",
                        "full_name": f"{cluster_emoji} {cluster_name}",
                        "count": all_clusters[cluster_num],
                        "cluster_id": cluster_num,
                    }
                )

            chart_df = pd.DataFrame(chart_data)

            fig = px.bar(
                chart_df,
                x="short_name",
                y="count",
                title="📊 Rozkład wszystkich klastrów",
                labels={"short_name": "Klaster", "count": "Liczba osób"},
                color="count",
                color_continuous_scale=get_color_palette(),
                custom_data=["full_name", "count"],
            )

            # Customizuj tooltip
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Osób: %{customdata[1]}<extra></extra>"
            )

            # Dodaj linię dla aktualnego klastra
            try:
                cluster_position = list(all_clusters.index).index(cluster_id)
                fig.add_vline(
                    x=cluster_position,
                    line_dash="dash",
                    line_color="#EC4899",
                    line_width=3,
                    annotation_text="⬅️ JESTEŚ TUTAJ",
                    annotation_position="top",
                    annotation_font_size=14,
                    annotation_font_color="#EC4899",
                )
            except (ValueError, IndexError):
                pass

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="white"),
                title_font=dict(size=16, color="white"),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Porównanie Ty vs Grupa
            st.markdown("#### 🏆 Co łączy Twoją grupę?")

            # Najczęstsze wartości w klastrze
            top_age = cluster_df["age"].mode()[0] if len(cluster_df) > 0 else "N/A"
            top_edu = (
                cluster_df["edu_level"].mode()[0] if len(cluster_df) > 0 else "N/A"
            )
            top_animal = (
                cluster_df["fav_animals"].mode()[0] if len(cluster_df) > 0 else "N/A"
            )
            top_place = (
                cluster_df["fav_place"].mode()[0] if len(cluster_df) > 0 else "N/A"
            )

            # Procent zgodności z grupą
            age_match = (
                (cluster_df["age"] == age).sum() / len(cluster_df) * 100
                if len(cluster_df) > 0
                else 0
            )
            edu_match = (
                (cluster_df["edu_level"] == edu_level).sum() / len(cluster_df) * 100
                if len(cluster_df) > 0
                else 0
            )
            animal_match = (
                (cluster_df["fav_animals"] == fav_animals).sum() / len(cluster_df) * 100
                if len(cluster_df) > 0
                else 0
            )
            place_match = (
                (cluster_df["fav_place"] == fav_place).sum() / len(cluster_df) * 100
                if len(cluster_df) > 0
                else 0
            )

            st.markdown(
                f"""
            ##### 👤 TWOJE WYBORY:
            - 🎂 **Wiek**: {age} {'✅' if age == top_age else f'(grupa: {top_age})'} `{age_match:.0f}% zgodności`
            - 🎓 **Wykształcenie**: {edu_level} {'✅' if edu_level == top_edu else f'(grupa: {top_edu})'} `{edu_match:.0f}%`
            - 🐾 **Zwierzęta**: {fav_animals} {'✅' if fav_animals == top_animal else f'(grupa: {top_animal})'} `{animal_match:.0f}%`
            - 📍 **Miejsce**: {fav_place} {'✅' if fav_place == top_place else f'(grupa: {top_place})'} `{place_match:.0f}%`
            
            ---
            
            ##### 📊 NAJPOPULARNIEJSZE W GRUPIE:
            - 🎂 {top_age}
            - 🎓 {top_edu}
            - 🐾 {top_animal}
            - 📍 {top_place}
            """
            )

    # TAB 2: Wiek
    with tab2:
        st.markdown("### 🎂 Analiza wieku w grupie")

        col1, col2 = st.columns(2)

        with col1:
            # Histogram wieku
            age_counts = cluster_df["age"].value_counts()
            fig = px.bar(
                y=age_counts.index,
                x=age_counts.values,
                title="Rozkład wieku",
                labels={"x": "Liczba osób", "y": "Grupa wiekowa"},
                orientation="h",
                color=age_counts.values,
                color_continuous_scale=get_color_palette(),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart
            fig = px.pie(
                values=age_counts.values,
                names=age_counts.index,
                title="Proporcje wieku",
                color_discrete_sequence=get_color_palette(),
                hole=0.4,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Wykształcenie
    with tab3:
        st.markdown("### 🎓 Analiza wykształcenia w grupie")

        col1, col2 = st.columns(2)

        with col1:
            edu_counts = cluster_df["edu_level"].value_counts()
            fig = px.bar(
                x=edu_counts.index,
                y=edu_counts.values,
                title="Rozkład wykształcenia",
                labels={"x": "Poziom wykształcenia", "y": "Liczba osób"},
                color=edu_counts.values,
                color_continuous_scale=get_color_palette(),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sunburst - wykształcenie x wiek (bez None)
            sunburst_df = cluster_df[["edu_level", "age"]].dropna()

            if len(sunburst_df) > 0:
                fig = px.sunburst(
                    sunburst_df,
                    path=["edu_level", "age"],
                    title="Wykształcenie według wieku",
                    color_discrete_sequence=get_color_palette(),
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Brak danych do wyświetlenia")

    # TAB 4: Lubimy (zwierzęta)
    with tab4:
        st.markdown("### 🐾 Analiza ulubionych zwierząt")

        col1, col2 = st.columns(2)

        with col1:
            animal_counts = cluster_df["fav_animals"].value_counts().head(10)
            fig = px.bar(
                y=animal_counts.index,
                x=animal_counts.values,
                title="Top 10 Ulubionych zwierząt",
                labels={"x": "Liczba osób", "y": "Zwierzę"},
                orientation="h",
                color=animal_counts.values,
                color_continuous_scale=get_color_palette(),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart zwierząt
            fig = px.pie(
                values=animal_counts.values,
                names=animal_counts.index,
                title="Proporcje ulubionych zwierząt",
                color_discrete_sequence=get_color_palette(),
                hole=0.4,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 5: Czujemy się najlepiej (miejsca)
    with tab5:
        st.markdown("### 📍 Analiza ulubionych miejsc")

        col1, col2 = st.columns(2)

        with col1:
            place_counts = cluster_df["fav_place"].value_counts().head(10)
            fig = px.bar(
                y=place_counts.index,
                x=place_counts.values,
                title="Top 10 Ulubionych miejsc",
                labels={"x": "Liczba osób", "y": "Miejsce"},
                orientation="h",
                color=place_counts.values,
                color_continuous_scale=get_color_palette(),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sunburst - miejsca x zwierzęta (bez None)
            sunburst_df = cluster_df[["fav_place", "fav_animals"]].dropna()

            if len(sunburst_df) > 0:
                fig = px.sunburst(
                    sunburst_df,
                    path=["fav_place", "fav_animals"],
                    title="Miejsca według zwierząt",
                    color_discrete_sequence=get_color_palette(),
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Brak danych do wyświetlenia")

else:
    st.markdown("<br>", unsafe_allow_html=True)

    # Centrowanie
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("# 🌟 Znajdź Swoją Grupę! 🌟")
        st.markdown("### Odkryj ludzi o podobnych zainteresowaniach ✨")

        st.markdown("---")

        st.success("👈 **Zacznij od wypełnienia formularza po lewej stronie**")

        st.markdown(
            """
        <div style='text-align: center; padding: 20px;'>
            <p style='font-size: 1.2em;'>
                Odpowiedz na kilka pytań i kliknij przycisk,<br>
                a my znajdziemy Twoją idealną grupę! 🎯
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Krok po kroku
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**1️⃣ Wypełnij**\nformularz")
        with c2:
            st.warning("**2️⃣ Kliknij**\nprzycisk")
        with c3:
            st.success("**3️⃣ Odkryj**\ngrupę!")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 1.2em; color: #EC4899;'>💜 Stworzone z pasją dla Data Science 💜</p>
</div>
""",
    unsafe_allow_html=True,
)
