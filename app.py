# ============================================================
# 1. IMPORTS
# ============================================================
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================
PROJECT_NAME = "RASID"
PROJECT_TITLE = "AI-Powered Early Warning System for Supply Chain Disruption Risk"
PAGES = ["Overview", "Risk Monitoring", "Early Warning", "Insights"]

TARGET_COLUMN = "Risk_Level"
TIMESTAMP_COLUMN = "Timestamp"
SUPPLIER_COLUMN = "Supplier_ID"
RISK_ORDER = ["Low", "Medium", "High"]

COLORS = {
    "navy": "#1B365D",
    "cyan": "#0CB4CC",
    "logo_blue": "#0C3C84",
    "Low": "#24A56A",
    "Medium": "#F0A020",
    "High": "#E0474C",
    "background": "#F7F9FC",
    "white": "#FFFFFF",
    "border": "#E5EAF0",
    "text": "#243447",
    "muted": "#6B7280",
}

RISK_TINTS = {
    "Low": "#E6F6EE",
    "Medium": "#FEF3DC",
    "High": "#FDE9EA",
}

WARNING_MESSAGES = {
    "Low": (
        "✓ Low Risk",
        "The model classified the current conditions as Low Risk.",
    ),
    "Medium": (
        "⚠ Monitor Closely",
        "The model classified the current conditions as Medium Risk.",
    ),
    "High": (
        "⚠ Early Warning",
        "The model classified the current supply chain conditions as High Risk.",
    ),
}

WATCHLIST_COLUMNS = [
    TIMESTAMP_COLUMN,
    SUPPLIER_COLUMN,
    TARGET_COLUMN,
    "Lead_Time_Days",
    "Inventory_Level",
    "Port_Congestion",
]

MONITORING_COLUMNS = [
    TIMESTAMP_COLUMN,
    SUPPLIER_COLUMN,
    "Port_Congestion",
    "geopolitical_risk",
    "weather_severity",
    "Supplier_Reliability",
    "Inventory_Level",
    "Lead_Time_Days",
    TARGET_COLUMN,
]

# ============================================================
# 3. PROJECT PATHS
# ============================================================
BASE_PATH = Path(__file__).resolve().parent

DATA_PATH = BASE_PATH / "supply_20000.csv"
LOGO_PATH = BASE_PATH / "logo.jpeg"
MODEL_PATH = BASE_PATH / "artifacts" / "model.joblib"
MODEL_INFO_PATH = BASE_PATH / "artifacts" / "model_info.json"

# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=f"{PROJECT_NAME} | Supply Chain Risk Dashboard",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 5. CUSTOM CSS
# ============================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    .stApp {{ background-color: {COLORS['background']}; color: {COLORS['text']}; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}

    h1, h2, h3, h4 {{ color: {COLORS['navy']}; font-weight: 600; letter-spacing: -0.01em; }}
    p, li, span, label {{ color: {COLORS['text']}; }}

    section[data-testid="stSidebar"] {{
        background-color: {COLORS['white']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.25rem; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 0.5rem 0.75rem; border-radius: 10px; width: 100%; cursor: pointer;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {{ display: none; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: {COLORS['background']}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
        background: {COLORS['navy']};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p {{
        color: {COLORS['white']}; font-weight: 600;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
        border-color: {COLORS['border']};
    }}
    div[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-card_"] {{
        background: {COLORS['white']}; border: 1px solid {COLORS['border']}; border-radius: 18px;
        box-shadow: 0 2px 10px rgba(27, 54, 93, 0.06);
    }}

    .kpi-card, .info-card {{
        background: {COLORS['white']}; border: 1px solid {COLORS['border']};
        border-radius: 18px; padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(27, 54, 93, 0.06);
    }}
    .kpi-card {{ margin-bottom: 1.2rem; }}
    .kpi-label {{ color: {COLORS['muted']}; font-size: 0.9rem; font-weight: 500; }}
    .kpi-value {{ color: {COLORS['navy']}; font-size: 2.1rem; font-weight: 700; line-height: 1.2; }}
    .kpi-note {{ color: {COLORS['muted']}; font-size: 0.82rem; }}

    .page-title {{ color: {COLORS['navy']}; font-size: 2rem; font-weight: 700; line-height: 1.25; margin: 0; }}
    .page-description {{ color: {COLORS['muted']}; font-size: 1rem; margin-top: 0.35rem; max-width: 46rem; }}
    .section-title {{ color: {COLORS['navy']}; font-size: 1.2rem; font-weight: 600; margin: 1.6rem 0 0.2rem 0; }}
    .section-note {{ color: {COLORS['muted']}; font-size: 0.9rem; margin-bottom: 0.6rem; }}

    .flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin-top: 0.8rem; }}
    .flow-step {{
        background: {COLORS['background']}; border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 0.6rem 1rem; color: {COLORS['navy']}; font-weight: 500;
    }}
    .flow-arrow {{ color: {COLORS['cyan']}; font-weight: 700; font-size: 1.2rem; }}

    .result-card {{
        background: {COLORS['white']}; border: 1px solid {COLORS['border']};
        border-radius: 18px; padding: 1.4rem 1.6rem; text-align: center;
        box-shadow: 0 2px 10px rgba(27, 54, 93, 0.06);
    }}
    .result-label {{ color: {COLORS['muted']}; font-size: 0.95rem; }}
    .result-value {{ font-size: 3rem; font-weight: 700; letter-spacing: 0.04em; line-height: 1.2; }}
    .warning-box {{ border-radius: 14px; padding: 1rem 1.2rem; margin-top: 0.8rem; }}
    .warning-title {{ font-weight: 700; font-size: 1.05rem; color: {COLORS['text']}; }}
    .warning-text {{ color: {COLORS['text']}; }}

    .prob-row {{ display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; }}
    .prob-name {{ width: 4.5rem; font-weight: 500; }}
    .prob-track {{ flex: 1; background: {COLORS['border']}; border-radius: 999px; height: 12px; overflow: hidden; }}
    .prob-fill {{ height: 100%; border-radius: 999px; }}
    .prob-value {{ width: 3.5rem; text-align: right; font-weight: 600; color: {COLORS['navy']}; }}

    .stButton > button, .stFormSubmitButton > button {{
        background: {COLORS['navy']}; color: {COLORS['white']}; border: none;
        border-radius: 10px; padding: 0.6rem 1.6rem; font-weight: 600;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: #2B4F80; color: {COLORS['white']};
    }}
    .stButton > button p, .stFormSubmitButton > button p {{ color: {COLORS['white']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================
def pretty_name(column_name: str) -> str:
    return column_name.replace("_", " ").title()


def render_header(title: str, description: str) -> None:
    text_column, logo_column = st.columns([5, 1])

    with text_column:
        st.markdown(
            f'<div class="page-title">{title}</div>'
            f'<div class="page-description">{description}</div>',
            unsafe_allow_html=True,
        )

    with logo_column:
        st.image(str(LOGO_PATH), width=150)


def render_section_title(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if note:
        st.markdown(
            f'<div class="section-note">{note}</div>',
            unsafe_allow_html=True,
        )


def render_kpi_card(
    label: str,
    value: str,
    note: str = "",
    value_color: str = COLORS["navy"],
) -> None:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{value_color}">{value}</div>
            <div class="kpi-note">{note}&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_title(title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=title,
        subtitle=subtitle,
        anchor="start",
        offset=14,
    )


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart
        .properties(
            padding={
                "top": 16,
                "left": 12,
                "right": 12,
                "bottom": 12,
            }
        )
        .configure(background=COLORS["white"])
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=COLORS["text"],
            titleColor=COLORS["muted"],
            gridColor="#EEF2F6",
            domainColor="#C9D2DE",
            tickColor="#C9D2DE",
        )
        .configure_legend(
            labelColor=COLORS["text"],
            titleColor=COLORS["muted"],
        )
        .configure_title(
            color=COLORS["navy"],
            subtitleColor=COLORS["muted"],
            fontSize=17,
            subtitleFontSize=13,
            font="Inter, Segoe UI, sans-serif",
            subtitleFont="Inter, Segoe UI, sans-serif",
        )
    )


def risk_color_scale() -> alt.Scale:
    return alt.Scale(
        domain=RISK_ORDER,
        range=[COLORS[level] for level in RISK_ORDER],
    )


def card(name: str):
    return st.container(
        border=True,
        key=f"card_{name}",
    )


def show_table(
    table: pd.DataFrame,
    height: int = 350,
) -> None:

    table = table.copy()

    if TIMESTAMP_COLUMN in table.columns:
        table[TIMESTAMP_COLUMN] = table[
            TIMESTAMP_COLUMN
        ].dt.strftime("%Y-%m-%d")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=height,
    )


def stop_with_error(
    message: str,
    expected_path: Path,
) -> None:

    st.error(message)

    st.markdown(
        f"Expected file: `{expected_path.name}`"
    )

    st.code(str(expected_path))

    st.stop()


# ============================================================
# 7. DATASET LOADING
# ============================================================
@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:

    dataset = pd.read_csv(path)

    dataset[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataset[TIMESTAMP_COLUMN]
    )

    return dataset


# ============================================================
# 8. MODEL LOADING
# ============================================================
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


# ============================================================
# 9. MODEL INFORMATION LOADING
# ============================================================
def load_model_info(path: Path) -> dict:

    with open(
        path,
        encoding="utf-8",
    ) as info_file:

        return json.load(info_file)


# ============================================================
# 10. FILE VALIDATION
# ============================================================
if not DATA_PATH.exists():
    stop_with_error(
        "RASID could not find the dataset.",
        DATA_PATH,
    )

if not MODEL_PATH.exists():
    stop_with_error(
        "RASID could not find the trained model.",
        MODEL_PATH,
    )

if not MODEL_INFO_PATH.exists():
    stop_with_error(
        "RASID could not find the model information file.",
        MODEL_INFO_PATH,
    )

if not LOGO_PATH.exists():
    stop_with_error(
        "RASID could not find the logo.",
        LOGO_PATH,
    )


df = load_dataset(DATA_PATH)

trained_model = load_model(MODEL_PATH)

try:
    model_info = load_model_info(
        MODEL_INFO_PATH
    )

except json.JSONDecodeError as error:

    st.error(
        f"RASID could not read the model information file: {error}"
    )

    st.stop()


# ============================================================
# 11. MODEL CONFIGURATION
# ============================================================
model_features = model_info.get(
    "features"
)

window_size = int(
    model_info.get(
        "window_size",
        7,
    )
)

target_column = model_info.get(
    "target",
    TARGET_COLUMN,
)

label_map = model_info.get(
    "label_map"
)

if label_map is None:

    st.error(
        "model_info.json does not contain 'label_map'."
    )

    st.stop()


class_labels = {
    int(code): label
    for label, code in label_map.items()
}


missing_features = [
    feature
    for feature in model_features
    if feature not in df.columns
]

row_count, column_count = df.shape


# ============================================================
# 12. SIDEBAR
# ============================================================
with st.sidebar:

    st.image(
        str(LOGO_PATH),
        width="stretch",
    )

    st.markdown(
        f'<div style="font-weight:700;'
        f'color:{COLORS["navy"]};'
        f'font-size:1.3rem;">'
        f'{PROJECT_NAME}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    selected_page = st.radio(
        "Pages",
        PAGES,
        label_visibility="collapsed",
    )


# ============================================================
# 13. OVERVIEW
# ============================================================
risk_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .reindex(
        RISK_ORDER,
        fill_value=0,
    )
)


def render_overview() -> None:

    render_header(
        PROJECT_NAME,
        PROJECT_TITLE,
    )

    kpi_columns = st.columns(4)

    with kpi_columns[0]:

        render_kpi_card(
            "Dataset Records",
            f"{row_count:,}",
            f"{column_count} columns",
        )

    for column, level in zip(
        kpi_columns[1:],
        RISK_ORDER,
    ):

        with column:

            share = (
                risk_counts[level]
                / row_count
                * 100
            )

            render_kpi_card(
                f"{level} Risk",
                f"{risk_counts[level]:,}",
                f"{share:.1f}% of records",
                COLORS[level],
            )

    distribution = pd.DataFrame(
        {
            "Risk Level": risk_counts.index,
            "Records": risk_counts.values,
        }
    )

    distribution["Share"] = (
        distribution["Records"]
        / row_count
    )

    bars = (
        alt.Chart(distribution)
        .mark_bar(
            cornerRadiusEnd=6,
            size=26,
        )
        .encode(
            y=alt.Y(
                "Risk Level:N",
                sort=RISK_ORDER,
                title=None,
            ),
            x=alt.X(
                "Records:Q",
                title="Records",
            ),
            color=alt.Color(
                "Risk Level:N",
                scale=risk_color_scale(),
                legend=None,
            ),
            tooltip=[
                "Risk Level",
                alt.Tooltip(
                    "Records:Q",
                    format=",",
                ),
                alt.Tooltip(
                    "Share:Q",
                    format=".1%",
                ),
            ],
        )
    )

    labels = (
        bars
        .mark_text(
            align="left",
            dx=6,
        )
        .encode(
            text=alt.Text(
                "Records:Q",
                format=",",
            ),
            color=alt.value(
                COLORS["text"]
            ),
        )
    )

    with card("distribution"):

        title = chart_title(
            "Risk Distribution",
            "Number of records in each risk level",
        )

        st.altair_chart(
            style_chart(
                (bars + labels).properties(
                    height=210,
                    title=title,
                )
            ),
            width="stretch",
            theme=None,
        )


# ============================================================
# 14. RISK MONITORING
# ============================================================
def apply_monitoring_filters() -> pd.DataFrame:

    supplier_ids = sorted(
        df[SUPPLIER_COLUMN].unique()
    )

    first_date = (
        df[TIMESTAMP_COLUMN]
        .min()
        .date()
    )

    last_date = (
        df[TIMESTAMP_COLUMN]
        .max()
        .date()
    )

    supplier_column, risk_column, date_column = st.columns(3)

    supplier = supplier_column.selectbox(
        "Supplier",
        ["All"] + supplier_ids,
        format_func=lambda value:
        value
        if value == "All"
        else f"Supplier {value}",
    )

    risk_level = risk_column.selectbox(
        "Risk Level",
        ["All"] + RISK_ORDER,
    )

    date_range = date_column.date_input(
        "Date Range",
        value=(
            first_date,
            last_date,
        ),
        min_value=first_date,
        max_value=last_date,
    )

    filtered = df

    if supplier != "All":

        filtered = filtered[
            filtered[SUPPLIER_COLUMN]
            == supplier
        ]

    if risk_level != "All":

        filtered = filtered[
            filtered[TARGET_COLUMN]
            == risk_level
        ]

    if len(date_range) == 2:

        start_date, end_date = date_range

        filtered = filtered[
            filtered[TIMESTAMP_COLUMN]
            .dt.date
            .between(
                start_date,
                end_date,
            )
        ]

    return filtered


def render_risk_monitoring() -> None:

    render_header(
        "Risk Monitoring",
        "Explore supply chain risk records and identify patterns across suppliers and risk levels.",
    )

    filtered_df = apply_monitoring_filters()

    render_section_title(
        "Filtered Records",
        f"{len(filtered_df):,} records match the selected filters.",
    )

    if filtered_df.empty:

        st.info(
            "No records match the selected filters."
        )

        return

    show_table(
        filtered_df[MONITORING_COLUMNS]
    )

    weekly_counts = (
        filtered_df
        .set_index(TIMESTAMP_COLUMN)
        .groupby(
            [
                pd.Grouper(freq="W"),
                TARGET_COLUMN,
            ]
        )
        .size()
        .reset_index(
            name="Records"
        )
    )

    trend = (
        alt.Chart(weekly_counts)
        .mark_line(
            strokeWidth=2.5
        )
        .encode(
            x=alt.X(
                f"{TIMESTAMP_COLUMN}:T",
                title="Week",
            ),
            y=alt.Y(
                "Records:Q",
                title="Records per week",
            ),
            color=alt.Color(
                f"{TARGET_COLUMN}:N",
                scale=risk_color_scale(),
                title="Risk Level",
            ),
            tooltip=[
                f"{TIMESTAMP_COLUMN}:T",
                f"{TARGET_COLUMN}:N",
                "Records:Q",
            ],
        )
    )

    with card("trend"):

        title = chart_title(
            "Risk Trend",
            "Number of records per week, by risk level",
        )

        st.altair_chart(
            style_chart(
                trend.properties(
                    height=300,
                    title=title,
                )
            ),
            width="stretch",
            theme=None,
        )

    render_section_title(
        "Supplier Risk Overview",
        "Records per supplier for the selected filters.",
    )

    supplier_summary = (
        filtered_df
        .pivot_table(
            index=SUPPLIER_COLUMN,
            columns=TARGET_COLUMN,
            values=TIMESTAMP_COLUMN,
            aggfunc="count",
            fill_value=0,
        )
        .reindex(
            columns=RISK_ORDER,
            fill_value=0,
        )
    )

    supplier_summary.insert(
        0,
        "Records",
        supplier_summary.sum(axis=1),
    )

    supplier_summary = (
        supplier_summary
        .rename(
            columns={
                level: f"{level} Risk"
                for level in RISK_ORDER
            }
        )
        .reset_index()
        .rename(
            columns={
                SUPPLIER_COLUMN: "Supplier"
            }
        )
    )

    show_table(
        supplier_summary.sort_values(
            "High Risk",
            ascending=False,
        )
    )

    high_risk_records = filtered_df[
        filtered_df[TARGET_COLUMN]
        == "High"
    ]

    if not high_risk_records.empty:

        render_section_title(
            "High Risk Watchlist",
            f"{len(high_risk_records):,} High-risk records, most recent first.",
        )

        show_table(
            high_risk_records[
                WATCHLIST_COLUMNS
            ]
            .sort_values(
                TIMESTAMP_COLUMN,
                ascending=False,
            )
        )


# ============================================================
# 15. EARLY WARNING
# ============================================================

def get_seven_day_window(
    supplier_id,
    day_7_date,
):

    supplier_data = (
        df[
            df[SUPPLIER_COLUMN]
            == supplier_id
        ]
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    selected_date = pd.Timestamp(
        day_7_date
    )

    matching_rows = supplier_data[
        supplier_data[TIMESTAMP_COLUMN]
        == selected_date
    ]

    if matching_rows.empty:

        return None

    selected_index = matching_rows.index[0]

    start_index = (
        selected_index
        - window_size
        + 1
    )

    if start_index < 0:

        return None

    window = supplier_data.iloc[
        start_index:selected_index + 1
    ].copy()

    if len(window) != window_size:

        return None

    return window


def run_risk_assessment(
    window,
):

    input_data = (
        window[
            model_features
        ]
        .to_numpy()
    )

    model_input = input_data.reshape(
        1,
        -1,
    )

    predicted_code = int(
        trained_model.predict(
            model_input
        )[0]
    )

    predicted_label = class_labels[
        predicted_code
    ]

    probabilities = {}

    if hasattr(
        trained_model,
        "predict_proba",
    ):

        predicted_probabilities = (
            trained_model
            .predict_proba(
                model_input
            )[0]
        )

        for code, probability in zip(
            trained_model.classes_,
            predicted_probabilities,
        ):

            probabilities[
                class_labels[int(code)]
            ] = float(
                probability
            )

    return {
        "label": predicted_label,
        "probabilities": probabilities,
    }

def render_assessment_result(assessment):

    label = assessment["label"]
    probabilities = assessment["probabilities"]

    # ========================================================
    # MAIN RISK RESULT
    # ========================================================

    with st.container(border=True):

        st.subheader("Risk Assessment")

        if label == "High":

            st.error("HIGH")
            st.write("⚠ Early Warning")

        elif label == "Medium":

            st.warning("MEDIUM")
            st.write("⚠ Monitor Closely")

        else:

            st.success("LOW")
            st.write("✓ Low Risk")

    # ========================================================
    # RISK PROBABILITY
    # ========================================================

    st.subheader("Risk Probability")

    probability_columns = st.columns(3)

    for column, level in zip(
        probability_columns,
        RISK_ORDER,
    ):

        probability = probabilities.get(
            level,
            0,
        )

        with column:

            with st.container(border=True):

                st.metric(
                    label=f"{level} Risk",
                    value=f"{probability * 100:.0f}%",
                )

                st.progress(probability)

def render_early_warning() -> None:

    render_header(
        "Early Warning",
        "Use the latest 7 days of supply chain conditions to predict the next Risk Level.",
    )

    if missing_features:

        st.error(
            "These model features are missing from the dataset: "
            + ", ".join(
                missing_features
            )
        )

        return

    # --------------------------------------------------------
    # Supplier + Day 7 Date
    # --------------------------------------------------------

    supplier_column, date_column = st.columns(2)

    supplier_ids = sorted(
        df[SUPPLIER_COLUMN].unique()
    )

    selected_supplier = (
        supplier_column.selectbox(
            "Supplier",
            supplier_ids,
            format_func=lambda value:
            f"Supplier {value}",
        )
    )

    supplier_data = (
        df[
            df[SUPPLIER_COLUMN]
            == selected_supplier
        ]
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    first_valid_date = (
        supplier_data[
            TIMESTAMP_COLUMN
        ]
        .iloc[window_size - 1]
        .date()
    )

    last_valid_date = (
        supplier_data[
            TIMESTAMP_COLUMN
        ]
        .max()
        .date()
    )

    selected_date = date_column.date_input(
        "Day 7 Date",
        value=last_valid_date,
        min_value=first_valid_date,
        max_value=last_valid_date,
    )

    # --------------------------------------------------------
    # Get the 7-day window
    # --------------------------------------------------------

    window = get_seven_day_window(
        selected_supplier,
        selected_date,
    )

    if window is None:

        st.warning(
            "There are not enough previous days available for this date."
        )

        return

    # --------------------------------------------------------
    # Show 7-day input window
    # --------------------------------------------------------

    render_section_title(
        "7-Day Input Window",
        f"Day 7: {selected_date}",
    )

    display_columns = [
        TIMESTAMP_COLUMN
    ] + model_features

    show_table(
        window[
            display_columns
        ],
        height=350,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "Run Risk Assessment"
    ):

        assessment = run_risk_assessment(
            window
        )

        st.session_state[
            "assessment"
        ] = assessment

    assessment = st.session_state.get(
        "assessment"
    )

    if assessment:

        render_section_title(
            "Model Prediction"
        )

        render_assessment_result(
            assessment
        )


# ============================================================
# 16. MODEL INSIGHTS
# ============================================================

def render_insights() -> None:

    render_header(
        "Model Insights",
        "Feature importance and performance of the final trained model.",
    )

    feature_importance = (
        trained_model
        .feature_importances_
    )

    feature_names = []

    for day in range(
        1,
        window_size + 1,
    ):

        for feature in model_features:

            feature_names.append(
                f"{feature}_{day}"
            )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_importance,
        }
    )

    importance_df[
        "Original Feature"
    ] = (
        importance_df[
            "Feature"
        ]
        .str.rsplit(
            "_",
            n=1,
        )
        .str[0]
    )

    aggregated_importance = (
        importance_df
        .groupby(
            "Original Feature"
        )["Importance"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    importance_chart_data = (
        aggregated_importance
        .reset_index()
        .rename(
            columns={
                "Original Feature": "Feature"
            }
        )
    )

    chart = (
        alt.Chart(
            importance_chart_data
        )
        .mark_bar(
            cornerRadiusEnd=6,
            size=22,
        )
        .encode(
            y=alt.Y(
                "Feature:N",
                sort="-x",
                title=None,
            ),
            x=alt.X(
                "Importance:Q",
                title="Feature importance",
            ),
            tooltip=[
                "Feature",
                alt.Tooltip(
                    "Importance:Q",
                    format=".4f",
                ),
            ],
        )
    )

    with card("importance"):

        title = chart_title(
            "Feature Importance",
            "Aggregated importance across the 7-day input window",
        )

        st.altair_chart(
            style_chart(
                chart.properties(
                    height=340,
                    title=title,
                )
            ),
            width="stretch",
            theme=None,
        )

    st.caption(
        "Feature importance shows how strongly each feature "
        "contributed to the model's classification decisions. "
        "It does not represent causal impact."
    )

    render_section_title(
        "Model Performance"
    )

    metric_columns = st.columns(4)

    metrics = [
        (
            "Validation Accuracy",
            model_info.get(
                "validation_accuracy"
            ),
        ),
        (
            "Validation Macro F1",
            model_info.get(
                "validation_macro_f1"
            ),
        ),
        (
            "Test Accuracy",
            model_info.get(
                "test_accuracy"
            ),
        ),
        (
            "Test Macro F1",
            model_info.get(
                "test_macro_f1"
            ),
        ),
    ]

    for column, (
        label,
        value,
    ) in zip(
        metric_columns,
        metrics,
    ):

        with column:

            render_kpi_card(
                label,
                "Not available"
                if value is None
                else f"{value:.4f}",
            )


# ============================================================
# 17. PAGE ROUTING
# ============================================================

PAGE_RENDERERS = {
    "Overview": render_overview,
    "Risk Monitoring": render_risk_monitoring,
    "Early Warning": render_early_warning,
    "Insights": render_insights,
}

PAGE_RENDERERS[
    selected_page
]()


# ============================================================
# 18. FOOTER
# ============================================================

st.markdown(
    f"""
    <div style="
        text-align:center;
        color:{COLORS["muted"]};
        font-size:0.8rem;
        margin-top:3rem;
    ">
        {PROJECT_NAME}: {PROJECT_TITLE}
    </div>
    """,
    unsafe_allow_html=True,
)