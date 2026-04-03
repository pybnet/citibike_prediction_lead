import streamlit as st
import requests

# FastAPI URL inside container
FASTAPI_URL = "http://fastapi-app:8000"

# ---- Accessibility & French config ----
st.set_page_config(
    page_title="Vélos en libre-service - Prévision",
    page_icon="🚲",
    layout="centered"
)

# High-contrast, large-text accessible styles
st.markdown("""
    <style>
        /* Force light background on the whole app */
        .stApp {
            background-color: #ffffff !important;
        }

        .block-container {
            background-color: #ffffff !important;
        }

        /* Fix dark header/toolbar at the top */
        header[data-testid="stHeader"] {
            background-color: #ffffff !important;
        }
        [data-testid="stToolbar"] {
            background-color: #ffffff !important;
        }
        #MainMenu, footer {
            visibility: hidden;
        }

        /* All generic text */
        html, body, [class*="css"], .stMarkdown, .stText, label, p, span, div {
            color: #1a1a1a !important;
        }

        h1, h2, h3, h4 {
            color: #003d73 !important;
        }

        html, body, [class*="css"] {
            font-size: 18px !important;
        }

        /* ---- Selectbox fix ---- */
        /* The dropdown container */
        [data-testid="stSelectbox"] > div > div {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 2px solid #005ea2 !important;
        }
        /* The selected value text */
        [data-testid="stSelectbox"] span {
            color: #1a1a1a !important;
        }
        /* The dropdown arrow icon */
        [data-testid="stSelectbox"] svg {
            fill: #1a1a1a !important;
        }
        /* The open dropdown list */
        [data-testid="stSelectbox"] ul {
            background-color: #ffffff !important;
        }
        /* Each option in the list */
        [data-testid="stSelectbox"] li {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
        }
        [data-testid="stSelectbox"] li:hover {
            background-color: #e8f0fe !important;
            color: #003d73 !important;
        }

        /* ---- Number input fix ---- */
        [data-testid="stNumberInput"] input {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 2px solid #005ea2 !important;
            font-size: 1.1rem !important;
        }

        /* ---- Text input fix ---- */
        [data-testid="stTextInput"] input {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 2px solid #005ea2 !important;
            font-size: 1.1rem !important;
        }

        /* ---- Radio buttons ---- */
        .stRadio label {
            color: #1a1a1a !important;
            font-size: 1.1rem !important;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background-color: #005ea2 !important;
            color: #ffffff !important;
            font-size: 1.2rem !important;
            font-weight: bold !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            border: 3px solid #003d73 !important;
            min-height: 52px !important;
        }
        .stButton > button:hover, .stButton > button:focus {
            background-color: #003d73 !important;
            outline: 4px solid #ffcc00 !important;
        }

        /* Focus ring */
        *:focus {
            outline: 4px solid #ffcc00 !important;
            outline-offset: 2px !important;
        }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: #f0f4f8 !important;
            border: 2px solid #c0ccd8 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: bold !important;
            color: #1a1a1a !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #444444 !important;
        }

        /* Alerts */
        .stSuccess { background-color: #d4edda !important; border-left: 6px solid #155724 !important; color: #155724 !important; }
        .stWarning { background-color: #fff3cd !important; border-left: 6px solid #856404 !important; color: #856404 !important; }
        .stInfo    { background-color: #cce5ff !important; border-left: 6px solid #004085 !important; color: #004085 !important; }
        .stError   { background-color: #f8d7da !important; border-left: 6px solid #721c24 !important; color: #721c24 !important; }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🚲 Prévision de disponibilité des vélos")
st.markdown(
    "<p style='font-size:1.1rem;'>Consultez la disponibilité des vélos et des bornes "
    "dans une station à une heure donnée.</p>",
    unsafe_allow_html=True
)

# ---- Fetch station list ----
try:
    stations_resp = requests.get(f"{FASTAPI_URL}/stations")
    stations_resp.raise_for_status()
    stations = stations_resp.json().get("station_names", [])
except Exception as e:
    st.error(f"Impossible de charger les stations : {e}")
    stations = []

# ---- User inputs ----
st.markdown("### 🔧 Paramètres de recherche")

col1, col2 = st.columns(2)

with col1:
    hour = st.number_input(
        "Heure de la journée (0–23)",
        min_value=0,
        max_value=23,
        value=10,
        help="Entrez l'heure souhaitée au format 24h. Par exemple : 14 pour 14h00."
    )

with col2:
    action = st.radio(
        "Que souhaitez-vous faire ?",
        options=["🚲 Prendre un vélo", "🅿️ Déposer un vélo"],
        horizontal=True,
        help="Choisissez votre intention pour adapter la recommandation."
    )

if stations:
    station = st.selectbox(
        "Choisissez une station",
        stations,
        help="Sélectionnez la station la plus proche de votre position."
    )
else:
    station = st.text_input(
        "Identifiant court de la station",
        value="6233.04",
        help="Entrez l'identifiant de la station (ex : 6233.04)."
    )

st.markdown("---")

# ---- Predict button ----
if st.button("🔍 Lancer la prévision", use_container_width=True):
    payload = {"hour_selected": hour, "station_selected": station}

    try:
        response = requests.post(f"{FASTAPI_URL}/forecast", json=payload)

        if response.status_code == 200:
            data  = response.json()
            pred  = data.get("predicted_net_flow", 0)
            bikes = data.get("num_bikes_available", "?")
            docks = data.get("num_docks_available", "?")

            st.success("✅ Prévision récupérée avec succès !")

            st.markdown("### 📊 Résultats")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Flux net prédit",
                f"{pred:+.1f}",
                help="Positif = plus de vélos attendus. Négatif = plus de départs prévus."
            )
            m2.metric("🚲 Vélos disponibles maintenant", bikes)
            m3.metric("🅿️ Bornes disponibles maintenant", docks)

            st.markdown("---")

            # ---- Recommendation ----
            st.markdown("### 💡 Recommandation")

            if action == "🚲 Prendre un vélo":
                if bikes == "?" or int(bikes) == 0:
                    st.warning(
                        "⚠️ Aucun vélo disponible en ce moment à cette station. "
                        "Essayez une station voisine."
                    )
                elif pred < 0:
                    st.info(
                        f"📉 Des vélos devraient **quitter** cette station vers {hour}h00. "
                        f"Il y a actuellement **{bikes} vélo(s)** disponible(s) — partez vite !"
                    )
                else:
                    st.success(
                        f"✅ Des vélos devraient **arriver** à cette station vers {hour}h00. "
                        f"Il y a actuellement **{bikes} vélo(s)** disponible(s)."
                    )

            else:  # Déposer un vélo
                if docks == "?" or int(docks) == 0:
                    st.warning(
                        "⚠️ Aucune borne libre en ce moment à cette station. "
                        "Essayez une station voisine."
                    )
                elif pred > 0:
                    st.info(
                        f"📈 Des vélos sont attendus à cette station vers {hour}h00. "
                        f"Il reste **{docks} borne(s)** disponible(s) — déposez vite avant saturation !"
                    )
                else:
                    st.success(
                        f"✅ Des vélos devraient **partir** de cette station vers {hour}h00. "
                        f"Il y a **{docks} borne(s)** disponible(s) — bon moment pour déposer !"
                    )

            with st.expander("📋 Réponse complète de l'API"):
                st.json(data)

        else:
            st.error(f"Erreur {response.status_code} — impossible d'obtenir une prévision.")
            st.text(response.text)

    except Exception as e:
        st.error(f"Échec de l'appel à l'API : {e}")