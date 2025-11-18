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

# Initialize logging
logger = setup_logging(config.log_level)
app_logger = logging.getLogger("biodiversity_intel.app")


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

                # Display results
                app_logger.info(f"Analysis completed successfully for '{species_name}'")
                st.success(f"Analysis complete for {species_name}")

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Summary",
                    "🚨 Threats",
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
                    threats = result.get('threats', [])
                    if threats:
                        for i, threat in enumerate(threats, 1):
                            st.write(f"{i}. {threat}")
                    else:
                        st.info("No specific threats identified.")

                with tab3:
                    st.header("Data Sources")

                    st.subheader("IUCN Red List Data")
                    st.json(result.get('iucn_data', {}))

                    st.subheader("GBIF Occurrence Data")
                    st.json(result.get('gbif_data', {}))

                    if result.get('news_data'):
                        st.subheader("Conservation News")
                        st.json(result.get('news_data', []))

                with tab4:
                    st.header("Full Assessment Report")
                    st.markdown(result.get('report', 'No report generated.'))

                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=result.get('report', ''),
                        file_name=f"{species_name.replace(' ', '_')}_assessment.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                app_logger.error(f"Error during analysis for '{species_name}': {e}", exc_info=True)
                st.error(f"Error analyzing species: {str(e)}")
                st.exception(e)

    elif analyze_button:
        app_logger.warning("Analysis button clicked with no species name")
        st.warning("Please enter a species name.")

    # Example species
    st.markdown("---")
    st.subheader("Example Species")

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("🐅 Tiger (Panthera tigris)", use_container_width=True):
            st.rerun()

    with example_col2:
        if st.button("🐼 Giant Panda (Ailuropoda melanoleuca)", use_container_width=True):
            st.rerun()

    with example_col3:
        if st.button("🦍 Mountain Gorilla (Gorilla beringei)", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
