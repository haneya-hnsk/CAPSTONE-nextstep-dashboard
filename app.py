
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

# Page configuration
st.set_page_config(
    page_title="Job Market Analysis Dashboard",
    page_icon="💼",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stPlot {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<h1 class="main-header">💼 Job Market Analysis Dashboard</h1><br>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_csv("data_2026-05-12T11_15_20.905021.csv")

    # Data cleaning
    fix_cols = ["experience_level", "salary"]
    for col in fix_cols:
        df[col] = df[col].fillna("Not Displayed")
    df = df.dropna()

    # Extract numeric salary
    df["numeric_salary"] = df["salary"].apply(extract_numeric_salary)

    # Determine country from location
    df["country"] = df["location"].apply(determine_country)

    skill_mapping_id_to_en = {
        # Bahasa Indonesia ke Inggris
        "komunikasi": "Communication",
        "kemunikasi": "Communication",  # typo handling
        "manajemen": "Management",
        "kepemimpinan": "Leadership",
        "akuntansi": "Accounting",
        "kepatuhan": "Compliance",
        "manajemen rantai pasok": "Supply Chain Management",
        "pemasaran": "Marketing",
        "inggris": "English",
        "layanan pelanggan": "Customer Service",
        "layanan keuangan": "Financial Services",
        "onboarding": "Onboarding",
        "hukum": "Legal",
        "biaya": "Cost Management",
        "dukungan media": "Support Media",
        "keuangan": "Finance",
        "accountancy": "Accountancy",
        "otomasi kualitas": "Quality Automation",
        "penggajian": "Payroll",
        "digital": "Digital",
        "pemecahan waktu": "Problem Solving",
        "pemangku kepentingan": "Stakeholder Management",
        "kepemimpinan": "Leadership",
        "laporan": "Reporting",
        "kepatuhan": "Compliance",
        "bahasa inggris": "English",
        "inggris": "English",
        "kepemimpinan": "Leadership",
        "penjualan": "Sales",
    }

    def map_skill_to_english(skill: str) -> str:
        skill_clean = re.sub(r"[^a-zA-Z0-9\s]", "", skill).strip().lower()
        return skill_mapping_id_to_en.get(skill_clean, skill.strip())

    def process_skills_re(text):
        if pd.isna(text) or text == "":
            return ""
        skills_list = re.split(r",\s*", str(text))
        mapped_skills = [map_skill_to_english(s) for s in skills_list if s.strip()]
        return ", ".join(mapped_skills)

    df["extracted_skills"] = df["extracted_skills"].apply(process_skills_re)

    return df


def determine_country(location):
    """Determine country from location string"""
    if not isinstance(location, str):
        return "Unknown"

    location_lower = location.lower()

    # Singapore indicators
    singapore_keywords = [
        "singapore",
        "central region",
        "west region",
        "east region",
        "north region",
    ]
    if any(keyword in location_lower for keyword in singapore_keywords):
        return "Singapore"

    # Malaysia indicators
    malaysia_keywords = [
        "malaysia",
        "kuala lumpur",
        "selangor",
        "penang",
        "johor",
        "my",
        "petaling jaya",
    ]
    if any(keyword in location_lower for keyword in malaysia_keywords):
        return "Malaysia"

    # Philippines indicators
    philippines_keywords = [
        "philippines",
        "manila",
        "cebu",
        "makati",
        "taguig",
        "pasig",
    ]
    if any(keyword in location_lower for keyword in philippines_keywords):
        return "Philippines"

    # India indicators
    india_keywords = [
        "india",
        "mumbai",
        "bangalore",
        "delhi",
        "chennai",
        "hyderabad",
        "pune",
    ]
    if any(keyword in location_lower for keyword in india_keywords):
        return "India"

    # Default to Indonesia for unspecified or Indonesian locations
    return "Indonesia"


def extract_numeric_salary(salary_str):
    """Extract numeric salary from string"""
    if not isinstance(salary_str, str):
        return None

    SGD_TO_IDR_RATE = 11500
    MYR_TO_IDR_RATE = 3500
    PHP_TO_IDR_RATE = 280
    INR_TO_IDR_RATE = 190

    conversion_factor = 1.0
    original_str_lower = salary_str.lower()

    if "sgd" in original_str_lower or (
        original_str_lower.startswith("$") and "usd" not in original_str_lower
    ):
        conversion_factor = SGD_TO_IDR_RATE
    elif "myr" in original_str_lower or "rm" in original_str_lower:
        conversion_factor = MYR_TO_IDR_RATE
    elif "php" in original_str_lower or "₱" in original_str_lower:
        conversion_factor = PHP_TO_IDR_RATE
    elif "inr" in original_str_lower or "₹" in original_str_lower:
        conversion_factor = INR_TO_IDR_RATE

    cleaned_str = original_str_lower.replace("rp", "").replace("idr", "")
    cleaned_str = cleaned_str.replace("sgd", "").replace("myr", "").replace("rm", "")
    cleaned_str = cleaned_str.replace("php", "").replace("inr", "")
    cleaned_str = cleaned_str.replace("$", "").replace("₱", "").replace("₹", "")
    cleaned_str = cleaned_str.replace(".", "").replace(",", "")
    cleaned_str = cleaned_str.replace("jt", "000000").replace("juta", "000000")
    cleaned_str = cleaned_str.replace("per month", "").replace("per year", "").strip()

    matches = re.findall(r"\d+", cleaned_str)

    numeric_value = None
    if len(matches) == 2:
        try:
            numeric_value = (float(matches[0]) + float(matches[1])) / 2
        except ValueError:
            numeric_value = None
    elif len(matches) == 1:
        try:
            numeric_value = float(matches[0])
        except ValueError:
            numeric_value = None

    if numeric_value is not None:
        return numeric_value * conversion_factor
    return None


# Load data
df = load_data()

# Sidebar filters
st.sidebar.title(" Filters")

# Country filter
countries = ["All"] + sorted(df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Job Category filter
job_categories = ["All"] + sorted(df["job_category"].unique().tolist())
selected_category = st.sidebar.selectbox("Select Job Category", job_categories)

# Employment Type filter
employment_types = ["All"] + df["employment_type"].value_counts().index[:3].tolist()
selected_employment = st.sidebar.selectbox("Select Employment Type", employment_types)

# Salary Range filter
if st.sidebar.checkbox("Filter by Salary Range (IDR)"):
    df_salary = df.dropna(subset=["numeric_salary"])
    if not df_salary.empty:
        min_salary = int(df_salary["numeric_salary"].min())
        max_salary = int(df_salary["numeric_salary"].max() // 12)
        salary_range = st.sidebar.slider(
            "Salary Range (in millions IDR)",
            min_value=min_salary // 1_000_000,
            max_value=max_salary // 1_000_000 + 1,
            value=(min_salary // 1_000_000, max_salary // 1_000_000),
        )
        salary_range = (salary_range[0] * 1_000_000, salary_range[1] * 1_000_000)

# Filter data based on selection
filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["job_category"] == selected_category]
if selected_employment != "All":
    filtered_df = filtered_df[filtered_df["employment_type"] == selected_employment]
if "salary_range" in locals():
    filtered_df = filtered_df.dropna(subset=["numeric_salary"])
    filtered_df = filtered_df[
        (filtered_df["numeric_salary"] >= salary_range[0])
        & (filtered_df["numeric_salary"] <= salary_range[1])
    ]

# Main Dashboard Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Jobs", f"{len(filtered_df):,}")
with col2:
    st.metric("Job Categories", filtered_df["job_category"].nunique())
with col3:
    st.metric("Countries", filtered_df["country"].nunique())
with col4:
    st.metric("Locations", filtered_df["location"].nunique())
with col5:
    if len(filtered_df.dropna(subset=["numeric_salary"])) > 0:
        avg_salary = filtered_df.dropna(subset=["numeric_salary"])[
            "numeric_salary"
        ].mean()
        if avg_salary >= 1_000_000_000:
            st.metric("Avg Salary", f"Rp {avg_salary/1_000_000_000:.1f}B")
        elif avg_salary >= 1_000_000:
            st.metric("Avg Salary", f"Rp {avg_salary/1_000_000:.1f}M")
        else:
            st.metric("Avg Salary", f"Rp {avg_salary:,.0f}")
    else:
        st.metric("Avg Salary", "N/A")

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(
    [
        " Job Distribution",
        " Salary Analysis",
        " Location Analysis",
        " Skills Analysis",
    ]
)

with tab1:
    st.subheader("Job Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Job Category Distribution Pie Chart
        fig = px.pie(
            filtered_df,
            names="job_category",
            title="Job Category Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(
            hovertemplate="<b>Kategori:</b> %{label}<br><b>Persentase:</b> %{percent}<extra></extra>"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top 10 Job Categories Bar Chart
        job_counts = filtered_df["job_category"].value_counts().head(10)
        fig = px.bar(
            x=job_counts.values,
            y=job_counts.index,
            orientation="h",
            title="Top 10 Job Categories",
            labels={"x": "Number of Jobs", "y": "Job Category"},
            color=job_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Employment Type and Country Distribution
    col1, col2 = st.columns([1, 1])

    with col1:
        employment_counts = filtered_df["employment_type"].value_counts().head(5)
        fig = px.pie(
            values=employment_counts.values,
            names=employment_counts.index,
            title="Employment Type Distribution",
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


with tab2:
    st.subheader("Salary Analysis")

    # Salary statistics
    df_with_salary = filtered_df.dropna(subset=["numeric_salary"])

    if len(df_with_salary) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_sal = df_with_salary["numeric_salary"].mean()
            st.metric("Average Salary", f"Rp {avg_sal:,.0f}")
        with col2:
            med_sal = df_with_salary["numeric_salary"].median()
            st.metric("Median Salary", f"Rp {med_sal:,.0f}")
        with col3:
            min_sal = df_with_salary["numeric_salary"].min()
            st.metric("Min Salary", f"Rp {min_sal:,.0f}")
        with col4:
            max_sal = df_with_salary["numeric_salary"].max() / 12
            st.metric("Max Salary", f"Rp {max_sal:,.0f}")

        col1, col2 = st.columns(2)

        with col1:
            # Salary Distribution by Job Category
            avg_salary_by_category = (
                df_with_salary.groupby("job_category")["numeric_salary"]
                .mean()
                .sort_values(ascending=True)
            )

            fig = px.bar(
                x=avg_salary_by_category.values,
                y=avg_salary_by_category.index,
                orientation="h",
                title="Average Salary by Job Category (IDR)",
                labels={"x": "Average Salary (IDR)", "y": "Job Category"},
                color=avg_salary_by_category.values,
                color_continuous_scale="RdYlGn",
                text=avg_salary_by_category.values,
            )
            fig.update_traces(texttemplate="Rp %{text:,.0f}", textposition="outside")
            fig.update_layout(
                height=600,
                yaxis={"categoryorder": "total ascending"},
                xaxis_tickformat=",.0f",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Salary Distribution by Country
            avg_salary_by_country = (
                df_with_salary[df_with_salary["country"] != "India"]
                .groupby("country")["numeric_salary"]
                .mean()
                .sort_values(ascending=True)
            )

            fig = px.bar(
                x=avg_salary_by_country.values,
                y=avg_salary_by_country.index,
                orientation="h",
                title="Average Salary by Country (IDR)",
                labels={"x": "Average Salary (IDR)", "y": "Country"},
                color=avg_salary_by_country.values,
                color_continuous_scale="Blues",
                text=avg_salary_by_country.values,
            )
            fig.update_traces(texttemplate="Rp %{text:,.0f}", textposition="outside")
            fig.update_layout(
                height=600,
                yaxis={"categoryorder": "total ascending"},
                xaxis_tickformat=",.0f",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Salary Box Plot by Employment Type
        fig = px.box(
            df_with_salary,
            x="employment_type",
            y="numeric_salary",
            title="Salary Distribution by Employment Type",
            labels={
                "employment_type": "Employment Type",
                "numeric_salary": "Salary (IDR)",
            },
            color="employment_type",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Salary Histogram
        fig = px.histogram(
            df_with_salary,
            x="numeric_salary",
            nbins=30,
            title="Salary Distribution Histogram",
            labels={"numeric_salary": "Salary (IDR)", "count": "Number of Jobs"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No salary data available for the current filters.")

with tab3:
    st.subheader("Location Analysis")

    # Country Distribution Pie Chart
    st.subheader("Country Distribution")
    country_counts = filtered_df["country"].value_counts()
    if "India" in country_counts.index:
        country_counts["Singapore"] = (
            country_counts.get("Singapore", 0) + country_counts["India"]
        )
        country_counts = country_counts.drop("India")

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

    # Top 30 Locations
    st.subheader("Top 30 Job Locations")

    top_30_locations = filtered_df["location"].value_counts().head(30)

    fig = px.bar(
        x=top_30_locations.values,
        y=top_30_locations.index,
        orientation="h",
        title="Top 30 Job Locations",
        labels={"x": "Number of Jobs", "y": "Location"},
        color=top_30_locations.values,
        color_continuous_scale="Viridis",
        text=top_30_locations.values,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=800, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Location details by country
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
                labels={"x": "Number of Jobs", "y": "Location"},
                color=country_locations.values,
                color_continuous_scale="Viridis",
                text=country_locations.values,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # Location vs Job Category Heatmap
    st.subheader("Location vs Job Category Distribution")

    # Get top 15 locations and their job category distribution
    top_15_locations = filtered_df["location"].value_counts().head(15).index
    heatmap_data = filtered_df[filtered_df["location"].isin(top_15_locations)]

    if not heatmap_data.empty:
        pivot_table = pd.crosstab(
            heatmap_data["location"], heatmap_data["job_category"]
        )

        fig = px.imshow(
            pivot_table,
            title="Job Category Distribution Across Top Locations",
            labels={"x": "Job Category", "y": "Location", "color": "Number of Jobs"},
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Skills Analysis")

    # Word Cloud for Skills
    if "extracted_skills" in filtered_df.columns:
        st.subheader("Most In-Demand Skills")

        col1, col2 = st.columns([3, 2])

        with col1:
            # Generate word cloud with better preprocessing
            text = " ".join(filtered_df["extracted_skills"].dropna().astype(str))
            if text:
                # Clean text for word cloud
                text = text.replace(",", " ").replace(";", " ").replace("and", " ")

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
                ax.set_title(
                    "Skills Word Cloud for Selected Filter",
                    fontsize=16,
                    fontweight="bold",
                )
                st.pyplot(fig)
            else:
                st.warning("No skills data available for the current filters.")

        with col2:
            # Top Skills Analysis with better extraction
            if text:
                # Extract individual skills (assuming comma or space separated)
                skills_list = []
                for skills_text in filtered_df["extracted_skills"].dropna():
                    # Split by common delimiters
                    skills = re.split(r"[,;]|\s+and\s+|\s+&\s+", str(skills_text))
                    skills_list.extend(
                        [skill.strip().lower() for skill in skills if skill.strip()]
                    )

                skill_freq = Counter(skills_list).most_common(15)
                skill_df = pd.DataFrame(skill_freq, columns=["Skill", "Frequency"])

                fig = px.bar(
                    x=skill_df["Frequency"],
                    y=skill_df["Skill"],
                    orientation="h",
                    title="Top 15 Most Required Skills",
                    labels={"x": "Frequency", "y": "Skill"},
                    color=skill_df["Frequency"],
                    color_continuous_scale="Viridis",
                    text=skill_df["Frequency"],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    height=500, yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Skills by Job Category
    if selected_category != "All" and "extracted_skills" in filtered_df.columns:
        st.subheader(f"Top Skills for {selected_category}")

        category_skills = []
        for skills_text in filtered_df["extracted_skills"].dropna():
            skills = re.split(r"[,;]|\s+and\s+|\s+&\s+", str(skills_text))
            category_skills.extend(
                [skill.strip().lower() for skill in skills if skill.strip()]
            )

        category_skill_freq = Counter(category_skills).most_common(10)
        category_skill_df = pd.DataFrame(
            category_skill_freq, columns=["Skill", "Frequency"]
        )

        fig = px.bar(
            x=category_skill_df["Frequency"],
            y=category_skill_df["Skill"],
            orientation="h",
            title=f"Top 10 Skills in {selected_category}",
            labels={"x": "Frequency", "y": "Skill"},
            color=category_skill_df["Frequency"],
            color_continuous_scale="Viridis",
            text=category_skill_df["Frequency"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Skills Trend by Country
    if "extracted_skills" in filtered_df.columns and selected_country == "All":
        st.subheader("Top Skills by Country")

        countries_to_show = filtered_df["country"].value_counts().head(4).index
        country_skills_data = []

        for country in countries_to_show:
            country_df = filtered_df[filtered_df["country"] == country]
            country_skills = []
            for skills_text in country_df["extracted_skills"].dropna():
                skills = re.split(r"[,;]|\s+and\s+|\s+&\s+", str(skills_text))
                country_skills.extend(
                    [skill.strip().lower() for skill in skills if skill.strip()]
                )

            top_skills = Counter(country_skills).most_common(5)
            for skill, freq in top_skills:
                country_skills_data.append(
                    {"Country": country, "Skill": skill, "Frequency": freq}
                )

        if country_skills_data:
            country_skills_df = pd.DataFrame(country_skills_data)

            fig = px.bar(
                country_skills_df,
                x="Skill",
                y="Frequency",
                color="Country",
                title="Top 5 Skills by Country",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Data Source:** Job listings database")
with col2:
    st.markdown(f"**Last Updated:** {datetime.datetime.now().strftime('%B %d, %Y')}")
with col3:
    st.markdown(f"**Total Records:** {len(df):,}")

# Summary statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(" Current Selection Summary")

if selected_country != "All":
    st.sidebar.info(f" Country: **{selected_country}**")
if selected_category != "All":
    st.sidebar.info(f" Category: **{selected_category}**")
if selected_employment != "All":
    st.sidebar.info(f" Employment: **{selected_employment}**")

st.sidebar.metric("Filtered Jobs", f"{len(filtered_df):,}")

if len(filtered_df) > 0:
    st.sidebar.metric("Filter Percentage", f"{(len(filtered_df)/len(df)*100):.1f}%")

    # Quick stats
    if len(filtered_df.dropna(subset=["numeric_salary"])) > 0:
        avg = filtered_df.dropna(subset=["numeric_salary"])["numeric_salary"].mean()
        st.sidebar.metric("Avg Salary (Filtered)", f"Rp {avg:,.0f}")

# Reset filters button
if st.sidebar.button(" Reset All Filters"):
    st.experimental_rerun()
