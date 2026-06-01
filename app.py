import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.ticker as mticker
import datetime
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
from sqlalchemy import create_engine
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components


# Page configuration
st.set_page_config(
    page_title="Job Market Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stPlot {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @media (min-width: 992px) {
        button[data-baseweb="tab"] p {
            font-size: 1.3rem !important; /* Ukuran teks lebih besar untuk desktop */
            font-weight: bold !important;  /* Membuat teks lebih tegas */
        }
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-header">Job Market Analysis Dashboard</h1>',
    unsafe_allow_html=True,
)

MIN_REASONABLE_SALARY = 500_000
MAX_REASONABLE_SALARY = 500_000_000

def get_db_connection():
    """Create SQLAlchemy engine from secrets"""
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    dbname = st.secrets["postgres"]["dbname"]
    user = st.secrets["postgres"]["user"]
    password = st.secrets["postgres"]["password"]

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)

@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load and fully process data, cached to prevent re-computing on UI interactions"""
    query = """
        SELECT 
            id,
            title,
            company,
            location,
            skills,
            employment_type,
            salary_min,
            salary_max,
            original_currency,
            source,
            category,
            uploaded_at
        FROM available_job_features 
        WHERE uploaded_at >= NOW() - INTERVAL '20 days'
        ORDER BY uploaded_at DESC
    """

    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        engine.dispose()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return pd.DataFrame()

    if df.empty:
        return df

    # OPTIMIZATION 1: Vectorized Salary Processing
    df["salary_min_clean"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max_clean"] = pd.to_numeric(df["salary_max"], errors="coerce")

    # Filter reasonable salaries
    mask_min = (df["salary_min_clean"] >= MIN_REASONABLE_SALARY) & (df["salary_min_clean"] <= MAX_REASONABLE_SALARY)
    df.loc[~mask_min, "salary_min_clean"] = np.nan

    mask_max = (df["salary_max_clean"] >= MIN_REASONABLE_SALARY) & (df["salary_max_clean"] <= MAX_REASONABLE_SALARY)
    df.loc[~mask_max, "salary_max_clean"] = np.nan

    # Swap min and max if min > max using vectorized assignment
    swap_mask = (df["salary_min_clean"] > df["salary_max_clean"]) & df["salary_min_clean"].notna() & df["salary_max_clean"].notna()
    df.loc[swap_mask, ["salary_min_clean", "salary_max_clean"]] = df.loc[swap_mask, ["salary_max_clean", "salary_min_clean"]].values

    # Calculate average
    df["avg_salary"] = df[["salary_min_clean", "salary_max_clean"]].mean(axis=1)

    # OPTIMIZATION 2: Vectorized Country Determination
    location_lower = df["location"].str.lower().fillna("")
    
    conditions = [
        location_lower.str.contains('singapore|sg|central singapore', regex=True),
        location_lower.str.contains('malaysia|kuala lumpur|selangor|penang|johor|petaling jaya|shah alam|subang jaya', regex=True),
        location_lower.str.contains('philippines|manila|cebu|makati|taguig|pasig|quezon city', regex=True),
        location_lower.str.contains('india|mumbai|bangalore|delhi|chennai|hyderabad|pune|gurgaon|noida', regex=True),
        location_lower.str.contains('thailand|bangkok|chiang mai|phuket', regex=True),
        location_lower.str.contains('vietnam|ho chi minh|hanoi|da nang', regex=True)
    ]
    choices = ["Singapore", "Malaysia", "Philippines", "India", "Thailand", "Vietnam"]
    
    df["country"] = np.select(conditions, choices, default="Indonesia")
    df["job_category"] = df["category"]

    # Process Skills
    def clean_single_skill_text(text):
        if not isinstance(text, str):
            return []
        skills = re.split(r"[,;]|\s+and\s+|\s+&\s+|\s*/\s*", text)
        cleaned = []
        for skill in skills:
            skill = skill.strip().lower()
            if skill and len(skill) > 1 and not skill.isdigit():
                skill = re.sub(r"[^a-zA-Z0-9#+\-.]", "", skill)
                if skill:
                    cleaned.append(skill)
        return cleaned

    df["cleaned_skills"] = df["skills"].apply(clean_single_skill_text)
    
    return df

# Load and process data
with st.spinner("Loading and processing data from database..."):
    df = load_and_process_data()

if df.empty:
    st.error("No data could be loaded from the database.")
    st.stop()

# ========== SIDEBAR FILTERS ==========
st.sidebar.title("Filters")



# Country
countries = ["All"] + sorted(df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

# Category
categories = ["All"] + sorted(df["job_category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

# Employment Type
emp_types = ["All"] + sorted(df["employment_type"].dropna().unique().tolist())
selected_employment = st.sidebar.selectbox("Employment Type", emp_types)

# Source
sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Source", sources)

# Currency filter
if st.sidebar.checkbox("Filter by Original Currency"):
    currencies = ["All"] + sorted(df["original_currency"].dropna().unique().tolist())
    selected_currency = st.sidebar.selectbox("Original Currency", currencies)
else:
    selected_currency = "All"

# Salary range filter
df_valid_salary = df.dropna(subset=["avg_salary"])
use_salary_filter = st.sidebar.checkbox("Filter by Salary Range")

if use_salary_filter and not df_valid_salary.empty:
    min_sal = int(df_valid_salary["avg_salary"].min())
    max_sal = int(df_valid_salary["avg_salary"].max())

    salary_range = st.sidebar.slider(
        "Salary Range (Rp, in millions)",
        min_value=min_sal // 1_000_000,
        max_value=max_sal // 1_000_000 + 1,
        value=(min_sal // 1_000_000, max_sal // 1_000_000),
    )
    salary_range = (salary_range[0] * 1_000_000, salary_range[1] * 1_000_000)

# ========== APPLY FILTERS ==========
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["job_category"] == selected_category]
if selected_employment != "All":
    filtered_df = filtered_df[filtered_df["employment_type"] == selected_employment]
if selected_source != "All":
    filtered_df = filtered_df[filtered_df["source"] == selected_source]
if selected_currency != "All":
    filtered_df = filtered_df[filtered_df["original_currency"] == selected_currency]
if use_salary_filter and "salary_range" in locals():
    filtered_df = filtered_df.dropna(subset=["avg_salary"])
    filtered_df = filtered_df[
        (filtered_df["avg_salary"] >= salary_range[0])
        & (filtered_df["avg_salary"] <= salary_range[1])
    ]

# ========== MAIN METRICS ==========
st.subheader("Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Jobs", f"{len(filtered_df):,}")
with col2:
    st.metric("Companies", f"{filtered_df['company'].nunique():,}")
with col3:
    st.metric("Categories", filtered_df["job_category"].nunique())
with col4:
    st.metric("Countries", filtered_df["country"].nunique())
with col5:
    st.metric("Locations", filtered_df["location"].nunique())
with col6:
    filtered_with_salary = filtered_df.dropna(subset=["avg_salary"])
    if len(filtered_with_salary) > 0:
        med_salary = filtered_with_salary["avg_salary"].median()
        if med_salary >= 1_000_000_000:
            st.metric("Median Salary", f"Rp {med_salary/1_000_000_000:.1f}B")
        elif med_salary >= 1_000_000:
            st.metric("Median Salary", f"Rp {med_salary/1_000_000:.1f}M")
        else:
            st.metric("Median Salary", f"Rp {med_salary:,.0f}")
    else:
        st.metric("Median Salary", "N/A")

# ========== TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Job Distribution",
        "Salary Analysis",
        "Location Analysis",
        "Skills Analysis",
    ]
)

with tab1:
    st.subheader("Job Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            filtered_df,
            names="job_category",
            title="Job Category Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        job_counts = filtered_df["job_category"].value_counts().head(10)
        fig = px.bar(
            x=job_counts.values,
            y=job_counts.index,
            orientation="h",
            title="Top 10 Job Categories",
            labels={"x": "Jobs", "y": "Category"},
            color=job_counts.values,
            color_continuous_scale="Viridis",
            text=job_counts.values,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        emp_counts = filtered_df["employment_type"].value_counts().head(5)
        fig = px.pie(
            values=emp_counts.values,
            names=emp_counts.index,
            title="Employment Type",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_counts = filtered_df["country"].value_counts()
        fig = px.pie(
            values=country_counts.values,
            names=country_counts.index,
            title="Country Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        source_counts = filtered_df["source"].value_counts().head(5)
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Job Source",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        top_companies = filtered_df["company"].value_counts().head(10)
        fig = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation="h",
            title="Top 10 Companies Hiring",
            labels={"x": "Jobs", "y": "Company"},
            color=top_companies.values,
            color_continuous_scale="Viridis",
            text=top_companies.values,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        filtered_df["uploaded_at"] = pd.to_datetime(filtered_df["uploaded_at"]).dt.date
        daily_jobs = (
            filtered_df.groupby("uploaded_at").size().reset_index(name="count")
        )
        daily_jobs = daily_jobs.sort_values("uploaded_at")

        fig = px.line(
            daily_jobs,
            x="uploaded_at",
            y="count",
            title="Daily Job Postings (Last 20 Days)",
            labels={"uploaded_at": "Date", "count": "Jobs"},
            markers=True,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Salary Analysis")

    df_salary = filtered_df.dropna(subset=["avg_salary"])

    if len(df_salary) > 0:
        Q1 = df_salary["avg_salary"].quantile(0.25)
        Q3 = df_salary["avg_salary"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_salary[
            (df_salary["avg_salary"] >= lower) & (df_salary["avg_salary"] <= upper)
        ]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Average*", f"Rp {df_clean['avg_salary'].mean():,.0f}")
        with col2:
            st.metric("Median", f"Rp {df_salary['avg_salary'].median():,.0f}")
        with col3:
            st.metric("Min*", f"Rp {df_clean['avg_salary'].min():,.0f}")
        with col4:
            st.metric("Max*", f"Rp {df_clean['avg_salary'].max():,.0f}")
        with col5:
            st.metric("Jobs with Salary", f"{len(df_salary):,}")

        st.caption("*After removing outliers (IQR method)")

        outliers_count = len(df_salary) - len(df_clean)
        if outliers_count > 0:
            st.info(
                f"{outliers_count:,} outliers removed ({(outliers_count/len(df_salary)*100):.1f}%)"
            )

        col1, col2 = st.columns(2)

        with col1:
            avg_by_cat = (
                df_clean.groupby("job_category")["avg_salary"]
                .median()
                .sort_values(ascending=True)
                .head(15)
            )
            fig = px.bar(
                x=avg_by_cat.values,
                y=avg_by_cat.index,
                orientation="h",
                title="Median Salary by Category",
                labels={"x": "Median Salary (Rp)", "y": "Category"},
                color=avg_by_cat.values,
                color_continuous_scale="RdYlGn",
                text=avg_by_cat.values,
            )
            fig.update_traces(texttemplate="Rp %{text:,.0f}", textposition="outside")
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_by_country = (
                df_clean.groupby("country")["avg_salary"]
                .median()
                .sort_values(ascending=True)
            )
            fig = px.bar(
                x=avg_by_country.values,
                y=avg_by_country.index,
                orientation="h",
                title="Median Salary by Country",
                labels={"x": "Median Salary (Rp)", "y": "Country"},
                color=avg_by_country.values,
                color_continuous_scale="Blues",
                text=avg_by_country.values,
            )
            fig.update_traces(texttemplate="Rp %{text:,.0f}", textposition="outside")
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                df_clean,
                x="employment_type",
                y="avg_salary",
                title="Salary by Employment Type (Outliers Removed)",
                labels={"employment_type": "Type", "avg_salary": "Salary (Rp)"},
                color="employment_type",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df_clean,
                x="avg_salary",
                nbins=30,
                title="Salary Distribution (Outliers Removed)",
                labels={"avg_salary": "Salary (Rp)", "count": "Jobs"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig.add_vline(
                x=df_clean["avg_salary"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean",
            )
            fig.add_vline(
                x=df_salary["avg_salary"].median(),
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        currency_counts = df_salary["original_currency"].value_counts()
        if len(currency_counts) > 1:
            fig = px.pie(
                values=currency_counts.values,
                names=currency_counts.index,
                title="Original Currency Distribution (Before Standardization)",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No salary data available for the current filters.")

with tab3:
    st.subheader("Location Analysis")

    country_counts = filtered_df["country"].value_counts()
    fig = px.pie(
        values=country_counts.values,
        names=country_counts.index,
        title="Job Distribution by Country",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label+value")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 30 Job Locations")
    top_30 = filtered_df["location"].value_counts().head(30)
    fig = px.bar(
        x=top_30.values,
        y=top_30.index,
        orientation="h",
        title="Top 30 Locations",
        labels={"x": "Jobs", "y": "Location"},
        color=top_30.values,
        color_continuous_scale="Viridis",
        text=top_30.values,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    if selected_country != "All":
        st.subheader(f"Locations in {selected_country}")
        country_df = filtered_df[filtered_df["country"] == selected_country]
        country_locations = country_df["location"].value_counts().head(20)

        if not country_locations.empty:
            fig = px.bar(
                x=country_locations.values,
                y=country_locations.index,
                orientation="h",
                title=f"Top 20 Locations in {selected_country}",
                labels={"x": "Jobs", "y": "Location"},
                color=country_locations.values,
                color_continuous_scale="Viridis",
                text=country_locations.values,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Geo Visualization")
    @st.cache_data(ttl=3600)
    def load_coordinated_data():
        try:
            df_geo = pd.read_csv('location_coordinates.csv')
            # Langsung bersihkan baris yang tidak memiliki koordinat
            df_geo_clean = df_geo.dropna(subset=['latitude', 'longitude'])
            return df_geo_clean
        except Exception as e:
            return pd.DataFrame()

    # Memuat data geospasial
    df_coordinated = load_coordinated_data()

    if not df_coordinated.empty:
        # Mempersiapkan data untuk HeatMap secara efisien
        heat_data = df_coordinated[['latitude', 'longitude', 'count']].values.tolist()

        if heat_data:
            # Menghitung nilai tengah peta secara instan dari data yang sudah difilter
            avg_lat = df_coordinated['latitude'].mean()
            avg_lon = df_coordinated['longitude'].mean()

            # Membuat peta Folium tanpa memuat elemen dekoratif bawaan yang berat
            m = folium.Map(
                location=[avg_lat, avg_lon], 
                zoom_start=5,
                tiles="OpenStreetMap"
            )

            # Menambahkan HeatMap ke dalam objek peta
            HeatMap(heat_data).add_to(m)
            components.html(m._repr_html_(), height=500)
            # st_folium(m, width="100%", height=500, returned_objects=[], use_container_width=True)
        else:
            st.warning("No data with coordinates to display a heatmap.")
    else:
        st.warning("File location_coordinates.csv not found or empty.")

    st.subheader("Location vs Category")
    top_15 = filtered_df["location"].value_counts().head(15).index
    heatmap_data = filtered_df[filtered_df["location"].isin(top_15)]

    if not heatmap_data.empty:
        pivot = pd.crosstab(heatmap_data["location"], heatmap_data["job_category"])
        fig = px.imshow(
            pivot,
            title="Job Category Across Top Locations",
            labels={"x": "Category", "y": "Location", "color": "Jobs"},
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Skills Analysis")

    all_skills = []
    for skills_list in filtered_df["cleaned_skills"].dropna():
        all_skills.extend(skills_list)

    if all_skills:
        col1, col2 = st.columns([3, 2])

        with col1:
            text = " ".join(all_skills)
            custom_stopwords = set(STOPWORDS)
            custom_stopwords.update(
                [
                    "job",
                    "role",
                    "work",
                    "experience",
                    "ability",
                    "skill",
                    "etc",
                    "will",
                    "required",
                    "requirements",
                    "must",
                    "good",
                    "knowledge",
                    "strong",
                ]
            )

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=custom_stopwords,
                max_words=100,
                collocations=False,
                min_font_size=10,
            ).generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Skills Word Cloud", fontsize=16, fontweight="bold")
            st.pyplot(fig)

        with col2:
            skill_freq = Counter(all_skills).most_common(20)
            skill_df = pd.DataFrame(skill_freq, columns=["Skill", "Frequency"])

            fig = px.bar(
                x=skill_df["Frequency"],
                y=skill_df["Skill"],
                orientation="h",
                title="Top 20 Most Required Skills",
                labels={"x": "Frequency", "y": "Skill"},
                color=skill_df["Frequency"],
                color_continuous_scale="Viridis",
                text=skill_df["Frequency"],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No skills data available.")

    if selected_category != "All":
        st.subheader(f"Top Skills for {selected_category}")
        cat_skills = []
        for skills_list in filtered_df["cleaned_skills"].dropna():
            cat_skills.extend(skills_list)

        if cat_skills:
            cat_freq = Counter(cat_skills).most_common(15)
            cat_df = pd.DataFrame(cat_freq, columns=["Skill", "Frequency"])

            fig = px.bar(
                x=cat_df["Frequency"],
                y=cat_df["Skill"],
                orientation="h",
                title=f"Top 15 Skills in {selected_category}",
                labels={"x": "Frequency", "y": "Skill"},
                color=cat_df["Frequency"],
                color_continuous_scale="Viridis",
                text=cat_df["Frequency"],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    if selected_country == "All":
        st.subheader("Top Skills by Country")
        top_countries = filtered_df["country"].value_counts().head(4).index
        country_skills_data = []

        for country in top_countries:
            c_df = filtered_df[filtered_df["country"] == country]
            c_skills = []
            for sl in c_df["cleaned_skills"].dropna():
                c_skills.extend(sl)

            for skill, freq in Counter(c_skills).most_common(5):
                country_skills_data.append(
                    {"Country": country, "Skill": skill, "Frequency": freq}
                )

        if country_skills_data:
            cs_df = pd.DataFrame(country_skills_data)
            fig = px.bar(
                cs_df,
                x="Skill",
                y="Frequency",
                color="Country",
                title="Top 5 Skills by Country",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# ========== FOOTER ==========
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Source:** PostgreSQL")
with col2:
    st.markdown("**Table:** available_job_features")
with col3:
    st.markdown(f"**Updated:** {datetime.datetime.now().strftime('%b %d, %Y %H:%M')}")
with col4:
    st.markdown(f"**Records:** {len(df):,}")

# ========== SUMMARY ==========
st.sidebar.markdown("---")
st.sidebar.subheader("Current Selection")
if selected_country != "All":
    st.sidebar.info(f"{selected_country}")
if selected_category != "All":
    st.sidebar.info(f"{selected_category}")
if selected_employment != "All":
    st.sidebar.info(f"{selected_employment}")
st.sidebar.metric("Filtered Jobs", f"{len(filtered_df):,}")
if len(filtered_df) > 0:
    st.sidebar.metric("% of Total", f"{(len(filtered_df)/len(df)*100):.0f}%")

if st.sidebar.button("Reset All Filters"):
    st.rerun()
# Data quality info
total_raw = len(df)
valid_salary = df["avg_salary"].notna().sum()
st.sidebar.info(
    f"Total jobs: {total_raw:,}\n"
    f"With salary: {valid_salary:,} ({(valid_salary/total_raw*100):.0f}%)"
)
