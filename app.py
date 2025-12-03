"""
Streamlit Web Application

Interactive interface for querying species and viewing threat assessments.
"""

import streamlit as st
import asyncio
import logging
from biodiversity_intel.workflow import run_conservation_analysis
from biodiversity_intel.config import config, setup_logging
import json
import pandas as pd
import plotly.express as px
from email.utils import parsedate_to_datetime
from typing import Dict

# Initialize logging
logger = setup_logging(config.log_level)
app_logger = logging.getLogger("biodiversity_intel.app")


def parse_year_from_rfc822(date_str: str):
    """Parse RFC 822 date and extract year."""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.year
    except:
        return None


def prepare_temporal_timeline(result: Dict) -> pd.DataFrame:
    """Aggregate all data sources by year."""
    timeline_data = []

    # IUCN assessments
    iucn_data = result.get('iucn_data', {})
    for assessment in iucn_data.get('assessment_history', []):
        try:
            year = int(assessment['year_published'])
            timeline_data.append({
                'year': year,
                'source': 'IUCN',
                'count': 1,
                'details': assessment
            })
        except (ValueError, TypeError, KeyError):
            # Skip entries with invalid years
            continue

    # GBIF occurrences
    gbif_data = result.get('gbif_data', {})
    for year_str, count in gbif_data.get('temporal_distribution', {}).items():
        try:
            year = int(year_str)
            timeline_data.append({
                'year': year,
                'source': 'GBIF',
                'count': count,
                'details': {'occurrences': count}
            })
        except (ValueError, TypeError):
            # Skip entries with invalid years
            continue

    # News articles
    for article in result.get('news_data', []):
        if article.get('pub_date'):
            year = parse_year_from_rfc822(article['pub_date'])
            if year:
                timeline_data.append({
                    'year': int(year),
                    'source': 'News',
                    'count': 1,
                    'details': article
                })

    df = pd.DataFrame(timeline_data)

    # Ensure year column is integer type if dataframe is not empty
    if not df.empty and 'year' in df.columns:
        df['year'] = df['year'].astype(int)

    return df


def create_timeline_chart(df: pd.DataFrame, source_filter: str):
    """Create interactive Plotly histogram."""
    if source_filter == 'All':
        # Aggregate by year and source
        chart_df = df.groupby(['year', 'source']).agg({'count': 'sum'}).reset_index()
        fig = px.bar(
            chart_df,
            x='year',
            y='count',
            color='source',
            title='Data Sources Timeline',
            labels={'count': 'Number of Entries', 'year': 'Year'},
            barmode='group',
            color_discrete_map={'IUCN': '#E63946', 'GBIF': '#06A77D', 'News': '#457B9D'}
        )
    else:
        # Single source
        chart_df = df.groupby('year').agg({'count': 'sum'}).reset_index()
        color_map = {'IUCN': '#E63946', 'GBIF': '#06A77D', 'News': '#457B9D'}
        fig = px.bar(
            chart_df,
            x='year',
            y='count',
            title=f'{source_filter} Timeline',
            labels={'count': 'Number of Entries', 'year': 'Year'},
            color_discrete_sequence=[color_map.get(source_filter, '#457B9D')]
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Count",
        hovermode='x unified',
        height=400
    )

    return fig


def main():
    """Main Streamlit application."""
    app_logger.info("Starting Streamlit application")

    st.set_page_config(
        page_title="Biodiversity Conservation Intelligence",
        page_icon="🌿",
        layout="wide"
    )

    st.title("🌿 Biodiversity Conservation Intelligence System")
    st.markdown("""
    An AI-powered threat intelligence system for biodiversity conservation.
    Query a species to get automated threat assessments from multiple data sources.
    """)

    app_logger.debug(f"App configuration: Model={config.openai_model}, Log Level={config.log_level}")

    # Initialize session state for species name
    if 'species_name' not in st.session_state:
        st.session_state.species_name = ""

    # Initialize session state for analysis results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # Initialize session state for timeline filters
    if 'selected_source' not in st.session_state:
        st.session_state.selected_source = 'All'
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info(f"LLM Provider: OpenAI")
        st.info(f"Model: {config.openai_model}")

        st.header("About")
        st.markdown("""
        This system integrates:
        - IUCN Red List data
        - GBIF occurrence records
        - Conservation news

        Using multi-agent LLM reasoning to generate comprehensive threat assessments.
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        species_name = st.text_input(
            "Enter Species Scientific Name",
            value=st.session_state.species_name,
            placeholder="e.g., Panthera tigris",
            help="Enter the scientific name of the species you want to analyze"
        )

    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("🔍 Analyze Species", type="primary", use_container_width=True)

    # Analysis section
    if analyze_button and species_name:
        app_logger.info(f"User requested analysis for species: '{species_name}'")

        with st.spinner(f"Analyzing {species_name}..."):
            try:
                # Run analysis
                app_logger.debug(f"Starting workflow execution for '{species_name}'")
                result = asyncio.run(run_conservation_analysis(species_name))

                with open('trash_result.json', 'w') as f:
                    json.dump(result, f)

                # Store result in session state
                st.session_state.analysis_result = result
                st.session_state.analyzed_species = species_name

                app_logger.info(f"Analysis completed successfully for '{species_name}'")

            except Exception as e:
                app_logger.error(f"Error during analysis for '{species_name}': {e}", exc_info=True)
                st.error(f"Error analyzing species: {str(e)}")
                st.exception(e)

    elif analyze_button:
        app_logger.warning("Analysis button clicked with no species name")
        st.warning("Please enter a species name.")

    # Display results if available in session state
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        analyzed_species = st.session_state.get('analyzed_species', 'Unknown')

        st.success(f"Analysis complete for {analyzed_species}")

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Summary",
            "🚨 Threats",
            "🔬 Analysis",
            "📉 Anomaly Detection",
            "📈 Data Sources",
            "📄 Full Report"
        ])

        with tab1:
            st.header("Conservation Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Population Trend", result.get('population_trend', 'Unknown'))
            with col2:
                confidence = result.get('confidence_score', 0.0)
                st.metric("Confidence Score", f"{confidence:.2%}")
            with col3:
                warning = "⚠️ Yes" if result.get('early_warning') else "✅ No"
                st.metric("Early Warning", warning)

        with tab2:
            st.header("Identified Threats")

            # Use threat_details if available, otherwise fall back to legacy threats
            threat_details = result.get('threat_details', [])

            if threat_details:
                st.markdown(f"**Total Threats:** {len(threat_details)}")
                st.markdown("---")

                for i, threat in enumerate(threat_details, 1):
                    code = threat.get('code', 'N/A')
                    mapped_name = threat.get('mapped_name', threat.get('name', 'Unknown threat'))

                    # Create expandable section for each threat
                    with st.expander(f"**{i}. {mapped_name}**", expanded=(i <= 5)):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown("**Threat Code:**")
                            st.code(code)
                        with col2:
                            st.markdown("**IUCN Classification:**")
                            st.write(mapped_name)
            else:
                # Fallback to legacy threats format
                threats = result.get('threats', [])
                if threats:
                    st.warning("⚠️ Displaying legacy threat format (codes only)")
                    for i, threat in enumerate(threats, 1):
                        st.write(f"{i}. {threat}")
                else:
                    st.info("No specific threats identified.")

        with tab3:
            st.header("LLM Analysis")

            # Display the analysis from AnalysisAgent
            analysis = result.get('analysis', 'No analysis available.')

            if analysis and analysis != 'No analysis available.':
                st.markdown("### Conservation Biologist Assessment")
                st.markdown(analysis)

                # Download button for analysis
                st.download_button(
                    label="📥 Download Analysis",
                    data=analysis,
                    file_name=f"{analyzed_species.replace(' ', '_')}_analysis.txt",
                    mime="text/plain",
                    key="download_analysis"
                )
            else:
                st.info("Analysis not available in results.")

        with tab4:
            st.header("Time-Series Anomaly Detection")

            # Get anomaly detection results from state
            anomaly_data = result.get('anomaly_detection')

            if anomaly_data and anomaly_data.get('summary_stats'):
                # Display summary metrics
                st.subheader("📊 Summary Statistics")

                summary_stats = anomaly_data['summary_stats']
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Years Analyzed", summary_stats.get('num_years', 0))
                with col2:
                    st.metric("Total Occurrences", summary_stats.get('total_occurrences', 0))
                with col3:
                    trend_direction = summary_stats.get('trend_direction', 'Unknown')
                    trend_icon = "📈" if trend_direction == "increasing" else "📉" if trend_direction == "decreasing" else "➡️"
                    st.metric("Trend", f"{trend_icon} {trend_direction.title()}")
                with col4:
                    st.metric("Anomalies Found", anomaly_data.get('num_anomalies', 0))

                # Display year range and trend
                st.markdown(f"**Year Range:** {summary_stats.get('year_range', 'N/A')}")
                st.markdown(f"**Mean Annual Count:** {summary_stats.get('mean_count', 0):.2f}")
                st.markdown(f"**Trend Slope:** {summary_stats.get('trend_slope', 0):.4f}")

                st.markdown("---")

                # Display visualization
                st.subheader("📉 Occurrence Time Series with Anomalies")

                if anomaly_data.get('figure_json'):
                    # Deserialize Plotly figure from JSON
                    import plotly.io as pio
                    fig = pio.from_json(anomaly_data['figure_json'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Visualization not available")

                st.markdown("---")

                # Display anomaly episodes
                st.subheader("🚨 Detected Anomaly Episodes")

                episodes = anomaly_data.get('episodes', [])
                num_declines = anomaly_data.get('num_declines', 0)
                num_surges = anomaly_data.get('num_surges', 0)

                if episodes:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Decline Episodes", num_declines, delta=None, delta_color="inverse")
                    with col2:
                        st.metric("Surge Episodes", num_surges, delta=None, delta_color="normal")

                    st.markdown("---")

                    # Group episodes by type
                    decline_episodes = [ep for ep in episodes if ep.get('type') == 'decline']
                    surge_episodes = [ep for ep in episodes if ep.get('type') == 'surge']

                    # Display decline episodes
                    if decline_episodes:
                        st.markdown("### 📉 Decline Anomalies")
                        st.markdown("*Periods where observation counts were significantly LOWER than expected*")

                        for i, ep in enumerate(decline_episodes, 1):
                            year_span = f"{ep['start_year']}" if ep['start_year'] == ep['end_year'] else f"{ep['start_year']}-{ep['end_year']}"

                            with st.expander(f"**Episode {i}: {year_span}** (Duration: {ep['num_years']} year(s))", expanded=(i <= 3)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Years", year_span)
                                    st.metric("Duration", f"{ep['num_years']} year(s)")
                                with col2:
                                    st.metric("Max |Z-score|", f"{ep['max_abs_z']:.2f}")
                                    st.metric("Mean Residual", f"{ep['mean_residual']:.3f}")

                                st.warning("⚠️ **Interpretation:** Observation counts were significantly lower than model predictions. This may indicate reduced sampling effort, population decline, or geographic range shift.")

                    # Display surge episodes
                    if surge_episodes:
                        st.markdown("### 📈 Surge Anomalies")
                        st.markdown("*Periods where observation counts were significantly HIGHER than expected*")

                        for i, ep in enumerate(surge_episodes, 1):
                            year_span = f"{ep['start_year']}" if ep['start_year'] == ep['end_year'] else f"{ep['start_year']}-{ep['end_year']}"

                            with st.expander(f"**Episode {i}: {year_span}** (Duration: {ep['num_years']} year(s))", expanded=(i <= 3)):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Years", year_span)
                                    st.metric("Duration", f"{ep['num_years']} year(s)")
                                with col2:
                                    st.metric("Max |Z-score|", f"{ep['max_abs_z']:.2f}")
                                    st.metric("Mean Residual", f"{ep['mean_residual']:.3f}")

                                st.info("ℹ️ **Interpretation:** Observation counts were significantly higher than model predictions. This may indicate increased sampling effort, population growth, or improved data collection.")

                    st.markdown("---")

                    # Important limitations
                    st.subheader("⚠️ Important Limitations")
                    st.warning("""
                    **Please note the following when interpreting anomaly detection results:**

                    1. **Sampling Artifacts:** Anomalies may reflect changes in data collection effort, taxonomy updates, or database ingestion patterns rather than true ecological changes.

                    2. **Yearly Resolution:** The model operates at annual granularity and cannot detect within-year seasonal or sub-annual patterns.

                    3. **Context Dependency:** Forecasts depend on having sufficient historical context. Early years in the time series have less reliable predictions.

                    4. **Statistical vs. Biological Significance:** A statistically significant anomaly does not automatically indicate biological importance. Domain expertise is required for interpretation.

                    5. **Spatial Aggregation:** This analysis ignores spatial distribution. Geographic shifts in occurrence could appear as temporal anomalies.
                    """)

                else:
                    st.info("No anomaly episodes detected in the time series.")

            else:
                st.info("Anomaly detection was not performed. This may be because:")
                st.markdown("""
                - No GBIF temporal distribution data was available
                - The time series was too short for analysis
                - An error occurred during anomaly detection
                """)

        with tab5:
            st.header("Temporal Data Timeline")

            # Prepare timeline data
            timeline_df = prepare_temporal_timeline(result)

            if timeline_df.empty:
                st.info("No temporal data available")
            else:
                # Source filter
                col1, col2 = st.columns([3, 1])

                with col1:
                    all_sources = ['All'] + sorted(timeline_df['source'].unique().tolist())

                    # Get default index based on session state
                    default_index = 0
                    if st.session_state.selected_source in all_sources:
                        default_index = all_sources.index(st.session_state.selected_source)

                    selected_source = st.radio(
                        "Select Data Source",
                        all_sources,
                        index=default_index,
                        horizontal=True,
                        key='source_radio'
                    )

                    # Update session state when selection changes
                    st.session_state.selected_source = selected_source

                # Filter data
                if selected_source != 'All':
                    filtered_df = timeline_df[timeline_df['source'] == selected_source]
                else:
                    filtered_df = timeline_df

                # Create and display chart
                fig = create_timeline_chart(filtered_df, selected_source)
                st.plotly_chart(fig, use_container_width=True)

                # Year selector for details
                st.subheader("Detailed View by Year")
                years_available = sorted(filtered_df['year'].unique(), reverse=True)

                if years_available:
                    # Get default index based on session state
                    default_year_index = 0
                    if st.session_state.selected_year in years_available:
                        default_year_index = years_available.index(st.session_state.selected_year)

                    selected_year = st.selectbox(
                        "Select Year",
                        years_available,
                        index=default_year_index,
                        help="View all data entries for the selected year",
                        key='year_selectbox'
                    )

                    # Update session state when selection changes
                    st.session_state.selected_year = selected_year

                    # Display details for selected year
                    year_data = filtered_df[filtered_df['year'] == selected_year]

                    # Group by source
                    for source in year_data['source'].unique():
                        source_data = year_data[year_data['source'] == source]

                        with st.expander(f"{source} - {selected_year} ({len(source_data)} entries)", expanded=True):
                            for idx, row in source_data.iterrows():
                                st.json(row['details'])

        with tab6:
            st.header("Full Assessment Report")
            st.markdown(result.get('report', 'No report generated.'))

            # Download button
            st.download_button(
                label="📥 Download Report",
                data=result.get('report', ''),
                file_name=f"{analyzed_species.replace(' ', '_')}_assessment.md",
                mime="text/markdown",
                key="download_report"
            )

    # Example species
    st.markdown("---")
    st.subheader("Example Species")

    # Row 1: Mammals
    st.markdown("**Mammals**")
    example_row1_col1, example_row1_col2, example_row1_col3, example_row1_col4 = st.columns(4)

    with example_row1_col1:
        if st.button("🐅 Tiger (Panthera tigris)", use_container_width=True):
            st.session_state.species_name = "Panthera tigris"
            st.rerun()

    with example_row1_col2:
        if st.button("🐼 Giant Panda (Ailuropoda melanoleuca)", use_container_width=True):
            st.session_state.species_name = "Ailuropoda melanoleuca"
            st.rerun()

    with example_row1_col3:
        if st.button("🦍 Mountain Gorilla (Gorilla beringei)", use_container_width=True):
            st.session_state.species_name = "Gorilla beringei"
            st.rerun()

    with example_row1_col4:
        if st.button("🐘 African Elephant (Loxodonta africana)", use_container_width=True):
            st.session_state.species_name = "Loxodonta africana"
            st.rerun()

    # Row 2: Marine & Reptiles
    st.markdown("**Marine & Reptiles**")
    example_row2_col1, example_row2_col2, example_row2_col3, example_row2_col4 = st.columns(4)

    with example_row2_col1:
        if st.button("🐢 Hawksbill Turtle (Eretmochelys imbricata)", use_container_width=True):
            st.session_state.species_name = "Eretmochelys imbricata"
            st.rerun()

    with example_row2_col2:
        if st.button("🦈 Great White Shark (Carcharodon carcharias)", use_container_width=True):
            st.session_state.species_name = "Carcharodon carcharias"
            st.rerun()

    with example_row2_col3:
        if st.button("🐊 American Alligator (Alligator mississippiensis)", use_container_width=True):
            st.session_state.species_name = "Alligator mississippiensis"
            st.rerun()

    with example_row2_col4:
        if st.button("🐍 King Cobra (Ophiophagus hannah)", use_container_width=True):
            st.session_state.species_name = "Ophiophagus hannah"
            st.rerun()

    # Row 3: Birds & Amphibians
    st.markdown("**Birds & Amphibians**")
    example_row3_col1, example_row3_col2, example_row3_col3, example_row3_col4 = st.columns(4)

    with example_row3_col1:
        if st.button("🦅 Bald Eagle (Haliaeetus leucocephalus)", use_container_width=True):
            st.session_state.species_name = "Haliaeetus leucocephalus"
            st.rerun()

    with example_row3_col2:
        if st.button("🦜 Kakapo (Strigops habroptilus)", use_container_width=True):
            st.session_state.species_name = "Strigops habroptilus"
            st.rerun()

    with example_row3_col3:
        if st.button("🐸 Golden Poison Frog (Phyllobates terribilis)", use_container_width=True):
            st.session_state.species_name = "Phyllobates terribilis"
            st.rerun()

    with example_row3_col4:
        if st.button("🦎 Komodo Dragon (Varanus komodoensis)", use_container_width=True):
            st.session_state.species_name = "Varanus komodoensis"
            st.rerun()

if __name__ == "__main__":
    main()
