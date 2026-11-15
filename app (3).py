import os
import requests
from datetime import date

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


# =========================================================
# ORCHARDAI CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OrchardAI",
    page_icon="🍎",
    layout="wide"
)

MODEL_PATH = "orchard_model.pkl"
DATA_PATH = "tree_health_data.csv"

DRIVE_FOLDER = "/content/drive/MyDrive/OrchardAI_Submission"
DRIVE_ORCHARDS_PATH = os.path.join(DRIVE_FOLDER, "orchards.csv")

if os.path.exists("/content/drive"):
    ORCHARDS_PATH = (
        DRIVE_ORCHARDS_PATH
        if os.path.exists(DRIVE_ORCHARDS_PATH)
        else "orchards.csv"
    )
else:
    ORCHARDS_PATH = "orchards.csv"

OBSERVATION_COLUMNS = [
    "Tree ID",
    "Date",
    "Growth stage",
    "Leaf condition",
    "Leaf color",
    "Leaf damage",
    "Disease symptoms",
    "Disease symptoms present",
    "Fruit condition",
    "Notes"
]

LEAF_DAMAGE_MAP = {
    "None observed": 0,
    "Spots": 1,
    "Holes": 2,
    "Drying": 3
}

FEATURE_COLUMNS = [
    "air_temperature",
    "air_humidity",
    "precipitation",
    "soil_temperature",
    "soil_moisture",
    "leaf_damage",
    "yellow_leaves_pct",
    "disease_symptoms"
]

ORCHARD_COLUMNS = [
    "company",
    "district",
    "location",
    "area_ha",
    "trees",
    "varieties",
    "latitude",
    "longitude",
    "location_status"
]

# Translate known Russian values from an existing orchards.csv.
# Company names are intentionally preserved.
ORCHARD_TEXT_TRANSLATIONS = {
    "Енбекшиказахский": "Enbekshikazakh District",
    "Талгарский": "Talgar District",
    "Талгарский и Енбекшиказахский":
        "Talgar and Enbekshikazakh Districts",
    "Алматинская область": "Almaty Region",
    "Каражотинский сельский округ": "Karazhotin Rural District",
    "Точное расположение не подтверждено":
        "Exact location not verified",
    "Несколько районов": "Multiple districts",
    "Село Тургень": "Turgen Village",
    "Казахстан, улица Ы. Калдыбаева, 28А":
        "Kazakhstan Village, Y. Kaldybaeva Street, 28A",
    "Более 12 сортов": "More than 12 varieties",
    "Интенсивный сад": "Intensive orchard",
    "14 сортов": "14 varieties",
    "Не указано": "Not specified",
    "Координаты не подтверждены": "Coordinates not verified",
    "Координаты сада не подтверждены":
        "Orchard coordinates not verified",
    "Координаты подтверждены пользователем":
        "Coordinates confirmed by user",
    "Точное местоположение уточняется":
        "Exact location pending verification",
    "Точные координаты сада не подтверждены":
        "Orchard coordinates not verified",
    "Не уточнены": "Not specified",
    "Не уточнено": "Not specified",
    "Не уточнена": "Not specified",
    "Село Казахстан": "Kazakhstan Village",
    "Gala, Fuji, Golden Delicious и другие":
        "Gala, Fuji, Golden Delicious, and others"
}


# =========================================================
# DATA STORAGE
# =========================================================

def load_observations():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    try:
        data = pd.read_csv(DATA_PATH)

        for column in OBSERVATION_COLUMNS:
            if column not in data.columns:
                data[column] = ""

        return data[OBSERVATION_COLUMNS]

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    except Exception as error:
        st.error(f"Could not load saved observations: {error}")
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def save_observations(data):
    temp_path = DATA_PATH + ".tmp"
    data.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, DATA_PATH)


def create_initial_orchards():
    return pd.DataFrame([
        {
            "company": "Arnau Agro",
            "district": "Enbekshikazakh District",
            "location": "Karazhotin Rural District",
            "area_ha": 470,
            "trees": 350000,
            "varieties": "Gala, Fuji, Golden Delicious",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Apple Bee",
            "district": "Almaty Region",
            "location": "Exact location not verified",
            "area_ha": 120,
            "trees": None,
            "varieties": "More than 12 varieties",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Baimene",
            "district": "Talgar District",
            "location": "Exact location not verified",
            "area_ha": 110,
            "trees": None,
            "varieties": "Intensive orchard",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Amal Bio",
            "district": "Talgar and Enbekshikazakh Districts",
            "location": "Multiple districts",
            "area_ha": 150,
            "trees": 500000,
            "varieties": "14 varieties",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Turgen Apple",
            "district": "Enbekshikazakh District",
            "location": "Turgen Village",
            "area_ha": 17.5,
            "trees": None,
            "varieties": "Golden Delicious, Starkrimson",
            "latitude": None,
            "longitude": None,
            "location_status": "Coordinates not verified"
        },
        {
            "company": "Agros BT",
            "district": "Enbekshikazakh District",
            "location": "Kazakhstan Village, Y. Kaldybaeva Street, 28A",
            "area_ha": None,
            "trees": None,
            "varieties": "Not specified",
            "latitude": None,
            "longitude": None,
            "location_status": "Orchard coordinates not verified"
        }
    ], columns=ORCHARD_COLUMNS)


def translate_orchard_text(data):
    data = data.copy()

    for column in ["district", "location", "varieties", "location_status"]:
        if column in data.columns:
            data[column] = data[column].apply(
                lambda value: ORCHARD_TEXT_TRANSLATIONS.get(
                    str(value).strip(),
                    value
                ) if pd.notna(value) else value
            )

    return data


def load_orchards():
    if not os.path.exists(ORCHARDS_PATH):
        return create_initial_orchards()

    try:
        data = pd.read_csv(ORCHARDS_PATH)

        for column in ORCHARD_COLUMNS:
            if column not in data.columns:
                data[column] = None

        data = data[ORCHARD_COLUMNS]
        return translate_orchard_text(data)

    except pd.errors.EmptyDataError:
        return create_initial_orchards()

    except Exception as error:
        st.error(f"Could not load orchard database: {error}")
        return create_initial_orchards()


def save_orchards(data):
    folder = os.path.dirname(ORCHARDS_PATH)

    if folder:
        os.makedirs(folder, exist_ok=True)

    temp_path = ORCHARDS_PATH + ".tmp"
    data.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, ORCHARDS_PATH)


if "observations" not in st.session_state:
    st.session_state.observations = load_observations()

if "orchards" not in st.session_state:
    st.session_state.orchards = load_orchards()

if "selected_latitude" not in st.session_state:
    st.session_state.selected_latitude = None

if "selected_longitude" not in st.session_state:
    st.session_state.selected_longitude = None


# =========================================================

# =========================================================
# PREMIUM ORCHARDAI DESIGN
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: #0B1F16;
    color: #EAF3E9;
}

/* MAIN CONTENT */
[data-testid="stAppViewContainer"] {
    background: #0B1F16;
}

[data-testid="stHeader"] {
    background: rgba(11, 31, 22, 0.95);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102B20, #0A1D14);
    border-right: 1px solid #294B37;
}

[data-testid="stSidebar"] * {
    color: #E7F0E7;
}

/* BRANDING */
h1, h2, h3 {
    color: #D9F0D2 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

[data-testid="stSidebar"] h1 {
    color: #B8E69E !important;
}

/* TEXT */
p, label, .stMarkdown {
    color: #DCE9DD;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #193B2A, #142F22);
    border: 1px solid #315940;
    border-radius: 16px;
    padding: 19px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.16);
}

[data-testid="stMetricLabel"] {
    color: #B4CCB8 !important;
}

[data-testid="stMetricValue"] {
    color: #C5EBAE !important;
    font-weight: 700;
}

/* BUTTONS */
.stButton button,
.stDownloadButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #387B4E, #245D3A);
    color: #FFFFFF !important;
    border: 1px solid #65966A;
    border-radius: 11px;
    padding: 9px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton button:hover,
.stDownloadButton button:hover,
.stFormSubmitButton button:hover {
    background: #4C965D;
    border-color: #A5D58F;
    transform: translateY(-1px);
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: #183629;
    color: #F0F6EF;
    border: 1px solid #42664B;
    border-radius: 9px;
}

/* FORMS AND CONTAINERS */
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: #122B1F;
    border: 1px solid #31533B;
    border-radius: 15px;
    padding: 10px;
}

/* DIVIDERS */
hr {
    border-color: #35533D !important;
}

/* ALERTS */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* RADIO BUTTONS */
[data-testid="stRadio"] label {
    color: #E2EEE1;
}

/* DATAFRAME CONTAINER */
[data-testid="stDataFrame"] {
    border: 1px solid #365941;
    border-radius: 13px;
    overflow: hidden;
}

/* DOWNLOAD BUTTON */
.stDownloadButton {
    margin-top: 8px;
}

/* FOOTER */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #A9C6AC;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PREMIUM TABLE HELPER
# =========================================================

def show_styled_table(data, **kwargs):
    """Dark forest-green premium dataframe."""

    if data is None or data.empty:
        st.info("No data available.")
        return

    styled = (
        data.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#17452F"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "700"),
                    ("padding", "14px"),
                    ("border", "1px solid #376B4B"),
                    ("text-align", "left")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "12px"),
                    ("color", "#F0F7F0"),
                    ("border-bottom", "1px solid #365B45")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [
                    ("background-color", "#193B2B"),
                    ("color", "#F0F7F0")
                ]
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", "#224A35"),
                    ("color", "#FFFFFF")
                ]
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("background-color", "#193B2B")
                ]
            }
        ])
    )

    st.dataframe(styled, **kwargs)


# =========================================================

# UNIFIED TEXT COLOR START
st.markdown("""
<style>
/* One consistent color for all green text */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stSidebar"] *,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"],
.stApp .stCaption,
.stApp [data-testid="stCaptionContainer"] {
    color: #E6F0E7 !important;
}

/* Keep buttons readable */
.stApp .stButton button,
.stApp .stDownloadButton button,
.stApp .stFormSubmitButton button {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)
# UNIFIED TEXT COLOR END

# WEATHER DATA
# =========================================================

WEATHER_LOCATIONS = [
    {"district": "Balkhash District", "place": "Bakanas"},
    {"district": "Enbekshikazakh District", "place": "Esik"},
    {"district": "Zhambyl District", "place": "Uzynagash"},
    {"district": "Ili District", "place": "Otegen Batyr"},
    {"district": "Karasay District", "place": "Kaskelen"},
    {"district": "Kegen District", "place": "Kegen"},
    {"district": "Raiymbek District", "place": "Narynkol"},
    {"district": "Talgar District", "place": "Talgar"},
    {"district": "Uyghur District", "place": "Chundzha"},
    {"district": "Konaev City", "place": "Kapchagay"},
    {"district": "Alatau City", "place": "Alatau"}
]


@st.cache_data(ttl=1800, show_spinner=False)
def load_regional_weather():

    rows = []

    for location in WEATHER_LOCATIONS:

        district = location["district"]
        place = location["place"]

        try:
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": place,
                    "count": 10,
                    "language": "en",
                    "format": "json",
                    "countryCode": "KZ"
                },
                timeout=15
            )

            geo_response.raise_for_status()
            geo_results = geo_response.json().get("results", [])

            matches = [
                item for item in geo_results
                if item.get("country_code") == "KZ"
            ]

            if not matches:
                raise ValueError("Location not found")

            # Prefer a result associated with Almaty Region
            selected = next(
                (
                    item for item in matches
                    if "almaty" in str(item.get("admin1", "")).casefold()
                ),
                matches[0]
            )

            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": selected["latitude"],
                    "longitude": selected["longitude"],
                    "current": (
                        "temperature_2m,"
                        "relative_humidity_2m,"
                        "wind_speed_10m,"
                        "precipitation"
                    ),
                    "timezone": "Asia/Almaty"
                },
                timeout=15
            )

            weather_response.raise_for_status()
            current = weather_response.json()["current"]

            rows.append({
                "Region": district,
                "Reference settlement": place,
                "Temperature (°C)": current.get("temperature_2m"),
                "Humidity (%)": current.get("relative_humidity_2m"),
                "Wind (km/h)": current.get("wind_speed_10m"),
                "Precipitation (mm)": current.get("precipitation"),
                "Local time": current.get("time"),
                "Status": "Available"
            })

        except Exception:
            rows.append({
                "Region": district,
                "Reference settlement": place,
                "Temperature (°C)": None,
                "Humidity (%)": None,
                "Wind (km/h)": None,
                "Precipitation (mm)": None,
                "Local time": None,
                "Status": "Unavailable"
            })

    return pd.DataFrame(rows)


weather = {
    "Temperature": 2.9,
    "Humidity": 83,
    "Precipitation": 0.0,
    "Wind speed": 4.5,
    "Soil temperature": 2.1,
    "Soil moisture": 0.334
}

forecast = pd.DataFrame({
    "Date": [
        "Nov 15", "Nov 16", "Nov 17",
        "Nov 18", "Nov 19", "Nov 20", "Nov 21"
    ],
    "Min temperature": [
        -1.4, -5.7, -5.7, -1.6, -0.5, 1.2, 0.3
    ],
    "Max temperature": [
        4.5, 0.5, 0.6, 7.2, 7.7, 11.9, 13.9
    ],
    "Precipitation": [
        1.7, 8.4, 0, 0, 0, 0, 0
    ]
})


# =========================================================
# MACHINE LEARNING FUNCTIONS
# =========================================================

def create_prediction_data(
    air_temp,
    air_humidity,
    precipitation,
    soil_temp,
    soil_moisture,
    leaf_damage,
    disease
):
    return pd.DataFrame([{
        "air_temperature": air_temp,
        "air_humidity": air_humidity,
        "precipitation": precipitation,
        "soil_temperature": soil_temp,
        "soil_moisture": soil_moisture,
        "leaf_damage": leaf_damage,
        "yellow_leaves_pct": 0,
        "disease_symptoms": disease
    }], columns=FEATURE_COLUMNS)


def show_prediction(model, tree_data):
    prediction = model.predict(tree_data)[0]
    probabilities = model.predict_proba(tree_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Low risk":
        st.success("Predicted Risk: Low")
    elif prediction == "Medium risk":
        st.warning("Predicted Risk: Medium")
    else:
        st.error("Predicted Risk: High")

    st.subheader("Risk Probabilities")

    for label, probability in zip(model.classes_, probabilities):
        st.write(f"**{label}: {probability:.1%}**")
        st.progress(float(probability))

    st.caption(
        "DEMO ONLY — the model was trained on synthetic data. "
        "This is not a confirmed tree diagnosis."
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🍎 OrchardAI")
st.sidebar.caption("Smart Care for Healthy Orchards")


# ORCHARDAI_VISUALS_START
import streamlit as st

ORCHARD_PAGE_ART = {'Dashboard': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzEyM2MyNyIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNmNGQyN2EiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODZiOTZiIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzg2Yjk2YiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4NmI5NmIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'Weather': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzE2NGM2NSIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNkN2U5ZjMiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjNzljNWM0Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzc5YzVjNCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM3OWM1YzQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'Tree Health': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzI0NGMyZCIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNmMGM2NmIiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOWFjYjcyIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzlhY2I3MiIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM5YWNiNzIiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'Analytics': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzI2M2IzNSIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNkNmRjYTAiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjODNiOGEwIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzgzYjhhMCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4M2I4YTAiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'ML Prediction': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzI0M2I1MCIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNkOGQ2YTIiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOGJiNGM3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhiYjRjNyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4YmI0YzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'Reports': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzRiMzgyOCIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNmMWQ5YTEiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjYzRhMzZjIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iI2M0YTM2YyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiNjNGEzNmMiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA==', 'Orchard Map': 'CiAgICA8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwMDAgMzYwIj4KICAgICAgPGRlZnM+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzI0NGMzYSIvPgogICAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMTAyNTFjIi8+CiAgICAgICAgPC9saW5lYXJHcmFkaWVudD4KICAgICAgPC9kZWZzPgogICAgICA8cmVjdCB3aWR0aD0iMTAwMCIgaGVpZ2h0PSIzNjAiIGZpbGw9InVybCgjc2t5KSIvPgogICAgICA8Y2lyY2xlIGN4PSI4MjAiIGN5PSI4NSIgcj0iNDUiIGZpbGw9IiNlNGQ0OWIiIG9wYWNpdHk9Ii44NSIvPgogICAgICA8cGF0aCBkPSJNMCAyNDAgUTI1MCAxMjAgNTAwIDIzMCBUMTAwMCAyMTAgVjM2MCBIMFoiCiAgICAgICAgICAgIGZpbGw9IiMzNjVlM2MiLz4KICAgICAgPHBhdGggZD0iTTAgMjgwIFEzMDAgMTkwIDYwMCAyODAgVDEwMDAgMjUwIFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMWM0MjJkIi8+CiAgICAgIAogICAgICAgIDxyZWN0IHg9Ijk1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMjMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSIxMTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSIyMjUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjMwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjIwOCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIyNTMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjE4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMjQ2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iMzU1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM2MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSIzMzgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iMzgzIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM0OCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9IjM3NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9IjQ4NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0OTAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNDY4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjUxMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI0NzgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI1MDYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgICA8cmVjdCB4PSI2MTUiIHk9IjIxNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjY1IiBmaWxsPSIjNjA0NTJkIi8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjIwIiBjeT0iMjA1IiByPSI0MiIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjU5OCIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI2NDMiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjA4IiBjeT0iMTk1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNjM2IiBjeT0iMjE1IiByPSI1IiBmaWxsPSIjYzg0YjM4Ii8+CiAgICAgICAgCiAgICAgICAgPHJlY3QgeD0iNzQ1IiB5PSIyMTUiIHdpZHRoPSIxMCIgaGVpZ2h0PSI2NSIgZmlsbD0iIzYwNDUyZCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc1MCIgY3k9IjIwNSIgcj0iNDIiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI3MjgiIGN5PSIyMjAiIHI9IjI1IiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iNzczIiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjczOCIgY3k9IjE5NSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIDxjaXJjbGUgY3g9Ijc2NiIgY3k9IjIxNSIgcj0iNSIgZmlsbD0iI2M4NGIzOCIvPgogICAgICAgIAogICAgICAgIDxyZWN0IHg9Ijg3NSIgeT0iMjE1IiB3aWR0aD0iMTAiIGhlaWdodD0iNjUiIGZpbGw9IiM2MDQ1MmQiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4ODAiIGN5PSIyMDUiIHI9IjQyIiBmaWxsPSIjOGViYzc3Ii8+CiAgICAgICAgPGNpcmNsZSBjeD0iODU4IiBjeT0iMjIwIiByPSIyNSIgZmlsbD0iIzhlYmM3NyIvPgogICAgICAgIDxjaXJjbGUgY3g9IjkwMyIgY3k9IjIyMCIgcj0iMjUiIGZpbGw9IiM4ZWJjNzciLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4NjgiIGN5PSIxOTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICA8Y2lyY2xlIGN4PSI4OTYiIGN5PSIyMTUiIHI9IjUiIGZpbGw9IiNjODRiMzgiLz4KICAgICAgICAKICAgICAgPHBhdGggZD0iTTAgMzI1IFE1MDAgMjg1IDEwMDAgMzI1IFYzNjAgSDBaIgogICAgICAgICAgICBmaWxsPSIjMTQyZDIwIi8+CiAgICA8L3N2Zz4KICAgIA=='}

ORCHARD_PAGE_CONTENT = {
    "Dashboard": ("🍎", "Welcome to OrchardAI",
                  "Intelligent insights for healthier apple orchards."),
    "Weather": ("🌤️", "Regional Weather Monitoring",
                "Weather conditions across the Almaty region."),
    "Tree Health": ("🌳", "Tree Health Monitoring",
                    "Observe orchard conditions and tree health."),
    "Analytics": ("📊", "Orchard Analytics",
                  "Explore agricultural data and performance indicators."),
    "ML Prediction": ("🧠", "OrchardAI ML Prediction",
                      "Explore machine-learning-based predictions."),
    "Reports": ("📄", "OrchardAI Reports",
                "Organized reports and orchard observations."),
    "Orchard Map": ("🗺️", "Orchard Map",
                    "Explore orchard locations and regional information.")
}

def show_orchard_hero(page_name):
    emoji, heading, subtitle = ORCHARD_PAGE_CONTENT[page_name]
    art = ORCHARD_PAGE_ART[page_name]

    st.markdown(
        f"""
        <div class="orchard-hero"
             style="background-image:
             linear-gradient(90deg, rgba(8,25,17,.94),
             rgba(8,25,17,.30)),
             url('data:image/svg+xml;base64,{art}');">
            <div class="orchard-hero-content">
                <div class="orchard-hero-icon">{emoji}</div>
                <div class="orchard-hero-title">{heading}</div>
                <div class="orchard-hero-subtitle">{subtitle}</div>
                <div class="orchard-hero-tag">ORCHARDAI • SMART AGRICULTURE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
.orchard-hero {
    min-height: 245px;
    background-size: cover;
    background-position: center;
    border-radius: 18px;
    margin: 8px 0 28px 0;
    padding: 30px 36px;
    display: flex;
    align-items: center;
    border: 1px solid rgba(160,200,150,.22);
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
}
.orchard-hero-content {
    max-width: 650px;
}
.orchard-hero-icon {
    font-size: 38px;
    margin-bottom: 8px;
}
.orchard-hero-title {
    color: white;
    font-size: 31px;
    font-weight: 750;
    line-height: 1.2;
    margin-bottom: 10px;
}
.orchard-hero-subtitle {
    color: #e0eadf;
    font-size: 15px;
}
.orchard-hero-tag {
    color: #b7d6a7;
    font-size: 10px;
    letter-spacing: 2px;
    margin-top: 22px;
}
.orchard-brand {
    text-align: center;
    padding: 8px 0 20px 0;
}
.orchard-brand-icon {
    font-size: 43px;
}
.orchard-brand-name {
    color: #d7ead0;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="orchard-brand">
    <div class="orchard-brand-icon">🍎</div>
    <div class="orchard-brand-name">ORCHARDAI</div>
</div>
""", unsafe_allow_html=True)
# ORCHARDAI_VISUALS_END


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Weather",
        "Tree Health",
        "Analytics",
        "ML Prediction",
        "Reports",
        "Orchard Map"
    ]
)

st.sidebar.divider()
st.sidebar.caption("📍 Almaty, Kazakhstan")
st.sidebar.caption("OrchardAI v1.4")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    show_orchard_hero("Dashboard")
    st.write("AI-powered insights for healthier apple orchards.")

    st.subheader("Baseline Weather Snapshot (demo)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", "2.9 °C")
    c2.metric("Humidity", "83%")
    c3.metric("Precipitation", "0 mm")
    c4.metric("Soil Temperature", "2.1 °C")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("7-Day Weather Forecast")
        st.line_chart(
            forecast.set_index("Date")[
                ["Min temperature", "Max temperature"]
            ]
        )

    with right:
        st.subheader("OrchardAI Status")
        st.success("Weather Dataset: Ready")
        st.success("Baseline v1.0: Ready")
        st.success("Tree Health Template: Ready")
        st.success("CSV Storage: Ready")

        if os.path.exists(MODEL_PATH):
            st.success("ML Engine v1.0: Loaded (DEMO)")
        else:
            st.warning("ML Model: File not found")

    st.divider()
    st.subheader("Project Overview")
    st.write(
        "OrchardAI combines environmental data and tree observations "
        "to support apple orchard health assessment."
    )

    st.info(
        "The current ML model is a demonstration model "
        "trained on synthetic data."
    )

    st.metric(
        "Saved Tree Observations",
        len(st.session_state.observations)
    )


# =========================================================
# WEATHER
# =========================================================

elif page == "Weather":

    show_orchard_hero("Weather")

    st.write(
        "Current weather conditions across the districts "
        "and cities of Almaty Region."
    )

    if st.button("🔄 Refresh Weather Data"):
        load_regional_weather.clear()
        st.rerun()

    with st.spinner("Loading regional weather data..."):
        regional_weather = load_regional_weather()

    available = regional_weather[
        regional_weather["Status"] == "Available"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Regions Loaded",
        f"{len(available)} / {len(WEATHER_LOCATIONS)}"
    )

    if not available.empty:
        c2.metric(
            "Average Temperature",
            f'{available["Temperature (°C)"].mean():.1f} °C'
        )

        c3.metric(
            "Average Humidity",
            f'{available["Humidity (%)"].mean():.0f}%'
        )
    else:
        c2.metric("Average Temperature", "Unavailable")
        c3.metric("Average Humidity", "Unavailable")

    st.divider()

    st.subheader("Regional Temperature and Humidity")

    show_styled_table(
        regional_weather,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Regional Weather CSV",
        data=regional_weather.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="OrchardAI_Regional_Weather.csv",
        mime="text/csv"
    )

    st.caption(
        "Weather data provided by Open-Meteo. "
        "Values represent the reference settlements of each district, "
        "not every location within the district. "
        "Data is cached for up to 30 minutes."
    )

    st.divider()
    st.subheader("Previous 7-Day Forecast Snapshot")

    show_styled_table(
        forecast,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TREE HEALTH
# =========================================================

elif page == "Tree Health":

    show_orchard_hero("Tree Health")
    st.write("Add observations about apple trees.")

    with st.form("tree_form"):

        tree_id = st.text_input("Tree ID")

        growth_stage = st.selectbox(
            "Growth Stage",
            [
                "Dormant",
                "Bud development",
                "Flowering",
                "Fruit development",
                "Harvest",
                "Unknown"
            ]
        )

        leaf_condition = st.selectbox(
            "Leaf Condition",
            [
                "Healthy",
                "Moderate damage",
                "Severe damage",
                "No leaves",
                "Unknown"
            ]
        )

        leaf_color = st.selectbox(
            "Leaf Color",
            [
                "Green",
                "Yellow",
                "Brown",
                "Mixed",
                "No leaves",
                "Unknown"
            ]
        )

        leaf_damage = st.selectbox(
            "Leaf Damage",
            [
                "None observed",
                "Spots",
                "Holes",
                "Drying",
                "Unknown"
            ]
        )

        disease = st.text_input(
            "Disease Symptoms",
            placeholder="Describe visible symptoms"
        )

        disease_present = st.selectbox(
            "Are disease symptoms present?",
            ["Unknown", "No", "Yes"]
        )

        fruit = st.selectbox(
            "Fruit Condition",
            ["Healthy", "Damaged", "No fruit", "Unknown"]
        )

        notes = st.text_area("Additional Notes")

        submitted = st.form_submit_button("Save Tree Observation")

        if submitted:

            if not tree_id.strip():
                st.error("Please enter a Tree ID.")

            else:
                new_record = pd.DataFrame([{
                    "Tree ID": tree_id.strip(),
                    "Date": str(date.today()),
                    "Growth stage": growth_stage,
                    "Leaf condition": leaf_condition,
                    "Leaf color": leaf_color,
                    "Leaf damage": leaf_damage,
                    "Disease symptoms": disease,
                    "Disease symptoms present": disease_present,
                    "Fruit condition": fruit,
                    "Notes": notes
                }])

                updated_observations = pd.concat(
                    [st.session_state.observations, new_record],
                    ignore_index=True
                )

                try:
                    save_observations(updated_observations)
                    st.session_state.observations = updated_observations
                    st.success("Observation saved successfully to CSV!")

                except Exception as error:
                    st.error(f"Could not save observation: {error}")

    st.divider()
    st.subheader("Tree Observations")

    st.metric(
        "Total Saved Observations",
        len(st.session_state.observations)
    )

    show_styled_table(
        st.session_state.observations,
        use_container_width=True
    )

    if not st.session_state.observations.empty:
        st.download_button(
            "Download Tree Health Dataset",
            data=st.session_state.observations.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="OrchardAI_Tree_Health.csv",
            mime="text/csv"
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    show_orchard_hero("Analytics")

    st.info(
        "The current Baseline analyzes environmental conditions, "
        "not confirmed tree health."
    )

    st.subheader("Environmental Risk Indicators")

    freezing_days = (forecast["Min temperature"] <= 0).sum()
    rainy_days = (forecast["Precipitation"] > 0).sum()

    c1, c2 = st.columns(2)
    c1.metric("Potential Freezing Days", freezing_days)
    c2.metric("Days with Precipitation", rainy_days)

    st.subheader("Temperature Analysis")
    st.bar_chart(
        forecast.set_index("Date")[
            ["Min temperature", "Max temperature"]
        ]
    )

    st.divider()
    st.subheader("Tree Observation Statistics")

    observations = st.session_state.observations
    st.metric("Total Tree Observations", len(observations))

    if not observations.empty:
        st.write("Observations by Leaf Condition")
        st.bar_chart(observations["Leaf condition"].value_counts())

        st.write("Observations by Growth Stage")
        st.bar_chart(observations["Growth stage"].value_counts())

    else:
        st.info("No tree observations have been saved yet.")


# =========================================================
# MACHINE LEARNING PREDICTION
# =========================================================

elif page == "ML Prediction":

    show_orchard_hero("ML Prediction")
    st.write("Machine Learning Engine v1.0")

    st.warning(
        "DEMO MODEL: trained on synthetic data. "
        "Predictions are not real tree diagnoses."
    )

    st.info(
        "The yellow-leaf percentage is no longer entered. "
        "The current model receives a fixed technical placeholder "
        "for compatibility."
    )

    if not os.path.exists(MODEL_PATH):

        st.error("ML model not found: orchard_model.pkl")
        st.info(
            "Upload orchard_model.pkl into the same "
            "Colab directory as app.py."
        )

    else:

        try:
            model = joblib.load(MODEL_PATH)
            st.success("ML Model successfully loaded!")

            analysis_mode = st.radio(
                "Choose Analysis Mode",
                ["Analyze Saved Tree", "Manual Analysis"]
            )

            # -------------------------------------------------
            # MODE 1: SAVED TREE
            # -------------------------------------------------

            if analysis_mode == "Analyze Saved Tree":

                st.subheader("Select Saved Tree")
                observations = st.session_state.observations.copy()

                if observations.empty:
                    st.info(
                        "No saved trees yet. "
                        "First add a tree in Tree Health."
                    )

                else:
                    observations["Leaf damage level"] = (
                        observations["Leaf damage"].map(LEAF_DAMAGE_MAP)
                    )

                    observations["Disease symptoms present"] = (
                        observations["Disease symptoms present"].fillna("")
                    )

                    valid_rows = observations[
                        observations["Leaf damage level"].notna()
                        & observations["Disease symptoms present"].isin(
                            ["Yes", "No"]
                        )
                    ].copy()

                    if valid_rows.empty:
                        st.warning(
                            "There are no observations with complete ML "
                            "inputs yet. Add a new observation in Tree Health. "
                            "Select a known Leaf Damage value and answer "
                            "Yes or No for disease symptoms."
                        )

                    else:
                        valid_rows["_row_number"] = valid_rows.index

                        selected_index = st.selectbox(
                            "Choose a tree observation",
                            valid_rows["_row_number"].tolist(),
                            format_func=lambda index: (
                                str(valid_rows.loc[index, "Tree ID"])
                                + " — "
                                + str(valid_rows.loc[index, "Date"])
                            )
                        )

                        selected_tree = valid_rows.loc[selected_index]

                        st.write("### Selected Tree Information")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Tree ID", selected_tree["Tree ID"])
                        c2.metric(
                            "Leaf Damage",
                            selected_tree["Leaf damage"]
                        )
                        c3.metric(
                            "Disease Symptoms",
                            selected_tree["Disease symptoms present"]
                        )

                        st.subheader("Environmental Data")
                        st.caption(
                            "Using the saved demonstration weather snapshot. "
                            "These are not historical measurements from the "
                            "tree's observation date."
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Air Temperature",
                            f'{weather["Temperature"]} °C'
                        )
                        c2.metric(
                            "Air Humidity",
                            f'{weather["Humidity"]}%'
                        )
                        c3.metric(
                            "Soil Moisture",
                            weather["Soil moisture"]
                        )

                        if st.button(
                            "🔍 Analyze Saved Tree",
                            type="primary"
                        ):
                            disease_value = (
                                1 if selected_tree[
                                    "Disease symptoms present"
                                ] == "Yes" else 0
                            )

                            tree_data = create_prediction_data(
                                weather["Temperature"],
                                weather["Humidity"],
                                weather["Precipitation"],
                                weather["Soil temperature"],
                                weather["Soil moisture"],
                                int(selected_tree["Leaf damage level"]),
                                disease_value
                            )

                            show_prediction(model, tree_data)

            # -------------------------------------------------
            # MODE 2: MANUAL ANALYSIS
            # -------------------------------------------------

            else:

                st.subheader("Enter Tree Data")
                col1, col2 = st.columns(2)

                with col1:
                    air_temp = st.number_input(
                        "Air Temperature (°C)",
                        value=2.9
                    )

                    air_humidity = st.number_input(
                        "Air Humidity (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=83.0
                    )

                    precipitation = st.number_input(
                        "Precipitation (mm)",
                        min_value=0.0,
                        value=0.0
                    )

                    soil_temp = st.number_input(
                        "Soil Temperature (°C)",
                        value=2.1
                    )

                with col2:
                    soil_moisture = st.number_input(
                        "Soil Moisture (m³/m³)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.334
                    )

                    leaf_damage = st.selectbox(
                        "Leaf Damage Level",
                        [0, 1, 2, 3]
                    )

                    disease = st.selectbox(
                        "Disease Symptoms",
                        [0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )

                st.divider()

                if st.button("🔍 Analyze Tree", type="primary"):
                    tree_data = create_prediction_data(
                        air_temp,
                        air_humidity,
                        precipitation,
                        soil_temp,
                        soil_moisture,
                        leaf_damage,
                        disease
                    )

                    show_prediction(model, tree_data)

        except Exception as error:
            st.error(f"Could not load or use ML model: {error}")


# =========================================================
# REPORTS
# =========================================================

elif page == "Reports":

    show_orchard_hero("Reports")
    st.write("Project data overview.")

    st.metric("Weather Records", "1")
    st.metric(
        "Tree Observations",
        len(st.session_state.observations)
    )
    st.metric("Forecast Period", "7 days")

    st.subheader("Export Data")

    st.download_button(
        "Download Weather Forecast CSV",
        forecast.to_csv(index=False).encode("utf-8-sig"),
        "OrchardAI_Weather.csv",
        "text/csv"
    )

    if not st.session_state.observations.empty:
        st.download_button(
            "Download Tree Health CSV",
            st.session_state.observations.to_csv(
                index=False
            ).encode("utf-8-sig"),
            "OrchardAI_Tree_Health.csv",
            "text/csv"
        )

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as data_file:
            st.download_button(
                "Download Saved Database",
                data_file,
                "tree_health_data.csv",
                "text/csv"
            )

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as model_file:
            st.download_button(
                "Download ML Model",
                model_file,
                "orchard_model.pkl",
                "application/octet-stream"
            )

    st.subheader("Orchard Database")
    st.metric(
        "Registered Orchards",
        len(st.session_state.orchards)
    )

    st.download_button(
        "Download Orchard Database",
        st.session_state.orchards.to_csv(
            index=False
        ).encode("utf-8-sig"),
        "orchards.csv",
        "text/csv"
    )


# =========================================================
# ORCHARD MAP
# =========================================================

elif page == "Orchard Map":

    show_orchard_hero("Orchard Map")
    st.write(
        "Explore registered apple orchards and approximate "
        "references in Almaty Region."
    )

    orchards = st.session_state.orchards.copy()

    # These are settlement/district reference points, NOT exact
    # orchard coordinates. Only use them as approximate map references.
    APPROXIMATE_LOCATIONS = {
        "arnau agro": {
            "latitude": 43.37113,
            "longitude": 77.45141,
            "reference": "Esik — approximate reference for Enbekshikazakh District"
        },
        "baimene": {
            "latitude": 43.30272,
            "longitude": 77.23947,
            "reference": "Talgar — approximate reference for Talgar District"
        },
        "turgen apple": {
            "latitude": 43.39672,
            "longitude": 77.59350,
            "reference": "Turgen Village — settlement reference"
        },
        "agros bt": {
            "latitude": 43.52000,
            "longitude": 77.77610,
            "reference": "Kazakhstan Village — settlement reference"
        }
    }

    def get_company_key(value):
        return str(value).strip().casefold()

    def has_coordinates(orchard):
        lat = pd.to_numeric(
            pd.Series([orchard.get("latitude")]),
            errors="coerce"
        ).iloc[0]

        lon = pd.to_numeric(
            pd.Series([orchard.get("longitude")]),
            errors="coerce"
        ).iloc[0]

        return pd.notna(lat) and pd.notna(lon)

    confirmed_count = 0
    approximate_count = 0
    not_mapped_count = 0

    for _, orchard in orchards.iterrows():
        if has_coordinates(orchard):
            confirmed_count += 1
        elif get_company_key(orchard["company"]) in APPROXIMATE_LOCATIONS:
            approximate_count += 1
        else:
            not_mapped_count += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Registered Orchards", len(orchards))
    c2.metric("Confirmed Locations", confirmed_count)
    c3.metric("Approximate References", approximate_count)

    st.info(
        "Orange markers show approximate settlement or district "
        "references, not the exact orchard location. Green markers "
        "show records with coordinates entered and confirmed by the user."
    )

    if not_mapped_count > 0:
        st.warning(
            f"{not_mapped_count} orchard record(s) do not have enough "
            "location information for a responsible map marker. "
            "They remain listed in the database below."
        )

    st.divider()

    map_mode = st.radio(
        "Map Type",
        ["Standard Map", "Satellite Map"],
        horizontal=True
    )

    orchard_map = folium.Map(
        location=[43.65, 77.40],
        zoom_start=7,
        tiles=None,
        control_scale=True
    )

    if map_mode == "Standard Map":
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Standard Map",
            control=False
        ).add_to(orchard_map)
    else:
        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Tiles © Esri",
            name="Satellite Map",
            control=False
        ).add_to(orchard_map)

    marker_points = []

    # Add records that contain coordinates.
    for _, orchard in orchards.iterrows():

        if not has_coordinates(orchard):
            continue

        latitude = float(orchard["latitude"])
        longitude = float(orchard["longitude"])
        status = str(orchard.get("location_status", ""))

        is_confirmed = (
            "confirmed by user" in status.casefold()
        )

        if is_confirmed:
            marker_color = "green"
            marker_label = "Coordinates confirmed by user"
        else:
            marker_color = "orange"
            marker_label = "Coordinates require verification"

        popup_text = (
            f"<b>{orchard['company']}</b><br>"
            f"District: {orchard['district']}<br>"
            f"Status: {marker_label}"
        )

        folium.Marker(
            location=[latitude, longitude],
            tooltip=str(orchard["company"]),
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(
                color=marker_color,
                icon="leaf",
                prefix="fa"
            )
        ).add_to(orchard_map)

        marker_points.append([latitude, longitude])

    # Add approximate settlement/district references for records
    # that do not have exact orchard coordinates.
    for _, orchard in orchards.iterrows():

        if has_coordinates(orchard):
            continue

        company_key = get_company_key(orchard["company"])
        reference_data = APPROXIMATE_LOCATIONS.get(company_key)

        if reference_data is None:
            continue

        latitude = reference_data["latitude"]
        longitude = reference_data["longitude"]

        popup_text = (
            f"<b>{orchard['company']}</b><br>"
            f"Approximate reference: {reference_data['reference']}<br>"
            "<b>Not the confirmed orchard location.</b>"
        )

        folium.Marker(
            location=[latitude, longitude],
            tooltip=f"{orchard['company']} — approximate reference",
            popup=folium.Popup(popup_text, max_width=350),
            icon=folium.Icon(
                color="orange",
                icon="info-sign",
                prefix="glyphicon"
            )
        ).add_to(orchard_map)

        marker_points.append([latitude, longitude])

    if marker_points:
        orchard_map.fit_bounds(marker_points)

    map_result = st_folium(
        orchard_map,
        width=None,
        height=580,
        returned_objects=["last_clicked"],
        key="orchard_map_widget"
    )

    if map_result and map_result.get("last_clicked"):
        clicked = map_result["last_clicked"]

        st.session_state.selected_latitude = round(
            clicked["lat"], 6
        )
        st.session_state.selected_longitude = round(
            clicked["lng"], 6
        )

    # ==========================================
    # REGIONAL WEATHER BELOW ORCHARD MAP
    # ==========================================

    st.divider()
    st.subheader("🌤 Regional Weather Monitoring")

    st.write(
        "Current temperature and humidity across "
        "the districts and cities of Almaty Region."
    )

    if st.button(
        "🔄 Refresh Regional Weather",
        key="map_weather_refresh"
    ):
        load_regional_weather.clear()
        st.rerun()

    with st.spinner("Loading regional weather..."):
        map_regional_weather = load_regional_weather()

    map_available = map_regional_weather[
        map_regional_weather["Status"] == "Available"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Locations Available",
        f"{len(map_available)} / {len(WEATHER_LOCATIONS)}"
    )

    if not map_available.empty:
        c2.metric(
            "Average Temperature",
            f'{map_available["Temperature (°C)"].mean():.1f} °C'
        )
        c3.metric(
            "Average Humidity",
            f'{map_available["Humidity (%)"].mean():.0f}%'
        )
    else:
        c2.metric("Average Temperature", "Unavailable")
        c3.metric("Average Humidity", "Unavailable")

    show_styled_table(
        map_regional_weather,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Weather data provided by Open-Meteo. "
        "Values represent reference settlements, not every location "
        "within each district. Data is cached for up to 30 minutes."
    )

    st.divider()
    st.subheader("📍 Register a New Orchard")

    st.write(
        "Click the actual orchard location on the map only "
        "if you know and can verify its position."
    )

    if (
        st.session_state.selected_latitude is not None
        and st.session_state.selected_longitude is not None
    ):
        st.success(
            "Selected coordinates: "
            f"{st.session_state.selected_latitude}, "
            f"{st.session_state.selected_longitude}"
        )
    else:
        st.info("No coordinates selected yet.")

    with st.form("new_orchard_form"):

        company = st.text_input("Company / Orchard Name")
        district = st.text_input("District")
        location = st.text_input("Village / Location Description")

        col1, col2 = st.columns(2)

        with col1:
            area_ha = st.number_input(
                "Orchard Area (ha)",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

            trees = st.number_input(
                "Number of Trees",
                min_value=0,
                value=0,
                step=100
            )

        with col2:
            varieties = st.text_input(
                "Apple Varieties",
                placeholder="Gala, Fuji, Golden Delicious"
            )

            coordinates_confirmed = st.checkbox(
                "I confirm that the selected map location "
                "is the actual orchard location"
            )

        submitted_orchard = st.form_submit_button("➕ Add Orchard")

        if submitted_orchard:

            if not company.strip():
                st.error("Enter the orchard or company name.")

            elif coordinates_confirmed and (
                st.session_state.selected_latitude is None
                or st.session_state.selected_longitude is None
            ):
                st.error(
                    "Select the actual orchard location on the map."
                )

            else:
                new_orchard = pd.DataFrame([{
                    "company": company.strip(),
                    "district": district.strip(),
                    "location": location.strip(),
                    "area_ha": area_ha if area_ha > 0 else None,
                    "trees": trees if trees > 0 else None,
                    "varieties": varieties.strip(),
                    "latitude": (
                        st.session_state.selected_latitude
                        if coordinates_confirmed else None
                    ),
                    "longitude": (
                        st.session_state.selected_longitude
                        if coordinates_confirmed else None
                    ),
                    "location_status": (
                        "Coordinates confirmed by user"
                        if coordinates_confirmed
                        else "Coordinates not verified"
                    )
                }])

                updated_orchards = pd.concat(
                    [st.session_state.orchards, new_orchard],
                    ignore_index=True
                )

                try:
                    save_orchards(updated_orchards)
                    st.session_state.orchards = updated_orchards

                    st.session_state.selected_latitude = None
                    st.session_state.selected_longitude = None

                    st.success(
                        f"{company.strip()} was successfully added "
                        "to the database!"
                    )
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not save orchard: {error}")

    st.divider()
    st.subheader("🌳 Registered Orchards Database")

    display_orchards = translate_orchard_text(
    st.session_state.orchards.copy()
)

    def get_map_status(row):
        if has_coordinates(row):
            return "Coordinates available"

        reference = APPROXIMATE_LOCATIONS.get(
            get_company_key(row["company"])
        )

        if reference:
            return "Approximate reference: " + reference["reference"]

        return "No verified map reference"

    display_orchards["Map Status"] = display_orchards.apply(
        get_map_status,
        axis=1
    )

    display_orchards = display_orchards.rename(columns={
        "company": "Company",
        "district": "District",
        "location": "Location",
        "area_ha": "Area (ha)",
        "trees": "Number of Trees",
        "varieties": "Apple Varieties",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "location_status": "Location Status"
    })

    show_styled_table(
        display_orchards,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Orchard Database CSV",
        data=st.session_state.orchards.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="orchards.csv",
        mime="text/csv"
    )

    st.caption(
        "Approximate markers represent nearby settlements or district "
        "references. They are not confirmed orchard locations."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("OrchardAI © 2026")
st.sidebar.caption("ML Engine v1.0 — DEMO")
