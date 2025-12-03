"""
Time-Series Anomaly Detection for GBIF Occurrence Data

Performs anomaly and change detection on species occurrence data using
a pretrained time-series forecasting model (Chronos) to identify unusual patterns.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import warnings
import logging
import hashlib
warnings.filterwarnings('ignore')

import torch
from chronos import ChronosPipeline
from biodiversity_intel.storage import FileCache
from biodiversity_intel.config import config

logger = logging.getLogger("biodiversity_intel.anomaly_detection")


class AnomalyEpisode(BaseModel):
    """Contiguous period of anomalous observations."""
    start_year: int
    end_year: int
    type: str  # 'decline' or 'surge'
    max_abs_z: float
    num_years: int
    mean_residual: float


class TimeSeriesStats(BaseModel):
    """Summary statistics for a time series."""
    scientific_name: str
    num_years: int
    year_range: str
    total_occurrences: int
    mean_count: float
    median_count: float
    min_count: int
    max_count: int
    trend_slope: float
    trend_direction: str


class AnomalyResults(BaseModel):
    """Complete anomaly detection results for a species."""
    species_name: str
    summary_stats: TimeSeriesStats
    num_anomalies: int
    num_declines: int
    num_surges: int
    episodes: List[AnomalyEpisode]


class GBIFAnomalyDetector:
    """
    Detects anomalies in GBIF occurrence time series using pretrained forecasting.
    Uses rolling backtesting where the model forecasts future values based on
    historical context, then compares predictions to actual observations.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        context_length: int = 12,
        forecast_horizon: int = 3,
        anomaly_threshold: float = 2.0,
        device: Optional[str] = None,
        enable_cache: Optional[bool] = None
    ):
        """
        Initialize detector with pretrained Chronos model.

        Args:
            model_size: Chronos model size (tiny, mini, small, base, large)
            context_length: Years of history to use for forecasting
            forecast_horizon: Years to forecast ahead
            anomaly_threshold: Z-score threshold for anomaly detection
            device: Device for model ('cpu', 'cuda', or None for auto-detect)
            enable_cache: Enable/disable result caching (None uses config default)
        """
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.anomaly_threshold = anomaly_threshold
        self.model_size = model_size

        # Initialize cache if enabled
        self.enable_cache = enable_cache if enable_cache is not None else config.enable_cache
        if self.enable_cache:
            self.cache = FileCache(cache_dir="data/cache/anomaly")
            logger.info(f"Initialized anomaly detector (caching enabled)")
        else:
            self.cache = None
            logger.info(f"Initialized anomaly detector (caching disabled)")

        # Auto-detect GPU availability if device not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.device = "cpu"
                logger.info("No GPU detected, using CPU")
        else:
            self.device = device

        # Load pretrained Chronos model with optimized settings
        logger.info(f"Loading Chronos-{model_size} model on {self.device}...")
        model_name = f"amazon/chronos-t5-{model_size}"

        # Use float16 for GPU to save memory, float32 for CPU
        if self.device == "cuda":
            torch_dtype = torch.float16  # Use float16 instead of bfloat16 for GTX 1650 Ti compatibility
        else:
            torch_dtype = torch.float32

        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
        )
        logger.info(f"Model loaded successfully from {model_name}")

        # Log GPU memory usage if using CUDA
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.debug(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def _get_cache_key(self, species_name: str) -> str:
        """
        Generate cache key for species anomaly detection results.
        Includes model parameters to ensure cache invalidation when settings change.
        """
        # Create a safe cache key from species name and model parameters
        safe_name = species_name.lower().replace(" ", "_").replace("/", "_")
        # Include model parameters in hash to invalidate cache when settings change
        params_str = f"{self.model_size}_{self.context_length}_{self.forecast_horizon}_{self.anomaly_threshold}"
        key_hash = hashlib.md5(f"{safe_name}_{params_str}".encode()).hexdigest()[:8]
        return f"anomaly_{safe_name}_{key_hash}"

    def parse_gbif_json(self, gbif_json: Dict) -> Tuple[str, pd.DataFrame]:
        """
        Parse GBIF JSON and create complete time series DataFrame.

        Args:
            gbif_json: Dictionary containing GBIF occurrence data

        Returns:
            Tuple of (scientific_name, time_series_dataframe)
        """
        scientific_name = gbif_json["data"]["scientific_name"]
        temporal_dist = gbif_json["data"]["temporal_distribution"]

        # Convert to DataFrame
        years = []
        counts = []
        for year_str, count in temporal_dist.items():
            try:
                year = int(year_str)
                years.append(year)
                counts.append(count)
            except ValueError:
                logger.warning(f"Skipping invalid year '{year_str}'")
                continue

        df = pd.DataFrame({
            'year': years,
            'count': counts
        }).sort_values('year').reset_index(drop=True)

        # Fill missing years with zeros
        if len(df) > 0:
            year_range = range(df['year'].min(), df['year'].max() + 1)
            df = df.set_index('year').reindex(year_range, fill_value=0).reset_index()
            df.columns = ['year', 'count']

        # Apply log transformation to stabilize variance
        # log1p(x) = log(1 + x) handles zeros gracefully
        df['log_count'] = np.log1p(df['count'])

        return scientific_name, df

    def rolling_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform rolling backtesting with pretrained forecasting model.
        For each time point, use historical context to forecast future values,
        then compare predictions with actual observations.

        Args:
            df: DataFrame with ['year', 'count', 'log_count'] columns

        Returns:
            DataFrame with predictions and residuals
        """
        results = []

        # Need context_length points before forecasting and forecast_horizon after for evaluation
        min_start = self.context_length
        max_start = len(df) - self.forecast_horizon

        if max_start < min_start:
            logger.warning(f"Time series too short for backtesting. "
                  f"Need at least {self.context_length + self.forecast_horizon} points, "
                  f"have {len(df)}")
            return pd.DataFrame()

        logger.info(f"Running rolling backtest from index {min_start} to {max_start}...")

        # Rolling window forecasting
        for t in range(min_start, max_start + 1):
            # Extract context window (historical data)
            context = df['log_count'].iloc[t - self.context_length:t].values

            # Prepare input tensor for Chronos
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            # Generate probabilistic forecast
            # Chronos returns multiple sample paths for uncertainty estimation
            with torch.no_grad():
                forecast_samples = self.pipeline.predict(
                    inputs=context_tensor,
                    prediction_length=self.forecast_horizon,
                    num_samples=100,
                ).numpy()

            # Clear GPU cache periodically to avoid memory issues
            if self.device == "cuda" and t % 10 == 0:
                torch.cuda.empty_cache()

            # Extract statistics from forecast samples
            # Shape: (1, num_samples, forecast_horizon) -> (num_samples, forecast_horizon)
            forecast_samples = forecast_samples[0]

            # For each horizon step, compare prediction with actual
            for h in range(self.forecast_horizon):
                actual_idx = t + h
                if actual_idx >= len(df):
                    break

                actual_year = df.loc[actual_idx, 'year']
                actual_log = df.loc[actual_idx, 'log_count']
                actual_count = df.loc[actual_idx, 'count']

                # Get forecast statistics for this horizon
                forecast_dist = forecast_samples[:, h]
                predicted_log = np.median(forecast_dist)
                pred_lower = np.percentile(forecast_dist, 10)
                pred_upper = np.percentile(forecast_dist, 90)
                pred_std = np.std(forecast_dist)

                results.append({
                    'year': int(actual_year),
                    'actual_count': int(actual_count),
                    'actual_log': actual_log,
                    'predicted_log': predicted_log,
                    'pred_lower': pred_lower,
                    'pred_upper': pred_upper,
                    'pred_std': pred_std,
                    'horizon': h + 1,
                    'context_start_year': int(df.loc[t - self.context_length, 'year'])
                })

        results_df = pd.DataFrame(results)

        # Aggregate predictions for each year
        # May have multiple predictions per year from different rolling windows
        # Use the prediction with smallest horizon (most recent)
        if len(results_df) > 0:
            results_df = results_df.sort_values(['year', 'horizon']).groupby('year').first().reset_index()

        return results_df

    def compute_anomaly_scores(self, backtest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute residuals and anomaly scores from backtest results.
        Anomalies are observations that deviate significantly from predictions.

        Args:
            backtest_df: DataFrame with actual and predicted values

        Returns:
            DataFrame with anomaly flags and scores
        """
        if len(backtest_df) == 0:
            return backtest_df

        # Compute residuals on log scale
        backtest_df['residual'] = backtest_df['actual_log'] - backtest_df['predicted_log']

        # Robust residual statistics using median and MAD
        residuals = backtest_df['residual'].values
        residual_median = np.median(residuals)
        mad = np.median(np.abs(residuals - residual_median))
        # Convert MAD to standard deviation equivalent for normal distribution
        residual_std = mad * 1.4826 if mad > 0 else np.std(residuals)

        # Compute z-scores
        backtest_df['z_score'] = (backtest_df['residual'] - residual_median) / residual_std

        # Flag anomalies
        backtest_df['anomaly_flag'] = np.abs(backtest_df['z_score']) >= self.anomaly_threshold

        # Classify anomaly type
        backtest_df['anomaly_type'] = backtest_df.apply(
            lambda row: (
                'decline' if row['anomaly_flag'] and row['z_score'] < 0
                else 'surge' if row['anomaly_flag'] and row['z_score'] > 0
                else 'normal'
            ),
            axis=1
        )

        logger.info(f"Anomaly detection summary:")
        logger.info(f"  Total forecasted points: {len(backtest_df)}")
        logger.info(f"  Anomalies detected: {backtest_df['anomaly_flag'].sum()}")
        logger.info(f"  Declines: {(backtest_df['anomaly_type'] == 'decline').sum()}")
        logger.info(f"  Surges: {(backtest_df['anomaly_type'] == 'surge').sum()}")

        return backtest_df

    def detect_episodes(self, anomaly_df: pd.DataFrame) -> List[AnomalyEpisode]:
        """
        Group contiguous anomalous years into episodes.
        An episode is a continuous sequence of years flagged as anomalous with same type.

        Args:
            anomaly_df: DataFrame with anomaly flags

        Returns:
            List of AnomalyEpisode objects
        """
        episodes = []

        # Filter to anomalous years only
        anomalous = anomaly_df[anomaly_df['anomaly_flag']].sort_values('year')

        if len(anomalous) == 0:
            return episodes

        # Group contiguous years
        current_episode = None

        for idx, row in anomalous.iterrows():
            year = row['year']
            anom_type = row['anomaly_type']
            z_score = row['z_score']
            residual = row['residual']

            if current_episode is None:
                # Start new episode
                current_episode = {
                    'start_year': year,
                    'end_year': year,
                    'type': anom_type,
                    'z_scores': [z_score],
                    'residuals': [residual]
                }
            elif (year == current_episode['end_year'] + 1 and
                  anom_type == current_episode['type']):
                # Extend current episode
                current_episode['end_year'] = year
                current_episode['z_scores'].append(z_score)
                current_episode['residuals'].append(residual)
            else:
                # Save previous episode and start new one
                episodes.append(AnomalyEpisode(
                    start_year=current_episode['start_year'],
                    end_year=current_episode['end_year'],
                    type=current_episode['type'],
                    max_abs_z=max(np.abs(current_episode['z_scores'])),
                    num_years=len(current_episode['z_scores']),
                    mean_residual=np.mean(current_episode['residuals'])
                ))
                current_episode = {
                    'start_year': year,
                    'end_year': year,
                    'type': anom_type,
                    'z_scores': [z_score],
                    'residuals': [residual]
                }

        # Save final episode
        if current_episode is not None:
            episodes.append(AnomalyEpisode(
                start_year=current_episode['start_year'],
                end_year=current_episode['end_year'],
                type=current_episode['type'],
                max_abs_z=max(np.abs(current_episode['z_scores'])),
                num_years=len(current_episode['z_scores']),
                mean_residual=np.mean(current_episode['residuals'])
            ))

        return episodes

    def compute_summary_stats(self, df: pd.DataFrame, species_name: str) -> TimeSeriesStats:
        """
        Compute basic summary statistics for time series.

        Args:
            df: Time series DataFrame
            species_name: Scientific name of species

        Returns:
            TimeSeriesStats object
        """
        # Linear trend using numpy polyfit
        if len(df) > 1:
            slope, intercept = np.polyfit(df['year'], df['count'], 1)
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
        else:
            slope = 0.0
            trend_direction = 'insufficient data'

        return TimeSeriesStats(
            scientific_name=species_name,
            num_years=len(df),
            year_range=f"{df['year'].min()}-{df['year'].max()}",
            total_occurrences=int(df['count'].sum()),
            mean_count=float(df['count'].mean()),
            median_count=float(df['count'].median()),
            min_count=int(df['count'].min()),
            max_count=int(df['count'].max()),
            trend_slope=float(slope),
            trend_direction=trend_direction
        )

    def visualize_results(
        self,
        df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        species_name: str,
        episodes: List[AnomalyEpisode]
    ) -> go.Figure:
        """
        Create interactive visualization of time series with anomalies.

        Args:
            df: Full time series DataFrame
            backtest_df: Backtest results with predictions
            species_name: Scientific name of species
            episodes: Detected anomaly episodes

        Returns:
            Plotly figure object
        """
        # Create subplots: main plot and residuals
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{species_name} - Occurrence Time Series with Anomaly Detection',
                'Forecast Residuals (Z-scores)'
            ),
            vertical_spacing=0.12
        )

        # Plot 1: Time series with predictions
        # Actual observations
        fig.add_trace(
            go.Scatter(
                x=df['year'],
                y=df['count'],
                mode='lines+markers',
                name='Actual Count',
                line=dict(color='blue', width=2),
                marker=dict(size=5)
            ),
            row=1, col=1
        )

        # Predicted values and anomalies
        if len(backtest_df) > 0:
            # Convert back from log scale
            backtest_df['predicted_count'] = np.expm1(backtest_df['predicted_log'])
            backtest_df['pred_lower_count'] = np.expm1(backtest_df['pred_lower'])
            backtest_df['pred_upper_count'] = np.expm1(backtest_df['pred_upper'])

            # Prediction line
            fig.add_trace(
                go.Scatter(
                    x=backtest_df['year'],
                    y=backtest_df['predicted_count'],
                    mode='lines',
                    name='Expected (Forecast)',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=1, col=1
            )

            # Prediction interval (80%)
            fig.add_trace(
                go.Scatter(
                    x=backtest_df['year'].tolist() + backtest_df['year'].tolist()[::-1],
                    y=backtest_df['pred_upper_count'].tolist() + backtest_df['pred_lower_count'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='80% Prediction Interval',
                    showlegend=True
                ),
                row=1, col=1
            )

            # Highlight decline anomalies
            decline_years = backtest_df[backtest_df['anomaly_type'] == 'decline']
            if len(decline_years) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=decline_years['year'],
                        y=decline_years['actual_count'],
                        mode='markers',
                        name='Decline Anomaly',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='x',
                            line=dict(width=2)
                        )
                    ),
                    row=1, col=1
                )

            # Highlight surge anomalies
            surge_years = backtest_df[backtest_df['anomaly_type'] == 'surge']
            if len(surge_years) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=surge_years['year'],
                        y=surge_years['actual_count'],
                        mode='markers',
                        name='Surge Anomaly',
                        marker=dict(
                            size=12,
                            color='green',
                            symbol='triangle-up',
                            line=dict(width=2)
                        )
                    ),
                    row=1, col=1
                )

        # Plot 2: Residual z-scores
        if len(backtest_df) > 0:
            colors = backtest_df['anomaly_type'].map({
                'normal': 'gray',
                'decline': 'red',
                'surge': 'green'
            })

            fig.add_trace(
                go.Bar(
                    x=backtest_df['year'],
                    y=backtest_df['z_score'],
                    name='Z-score',
                    marker=dict(color=colors),
                    showlegend=False
                ),
                row=2, col=1
            )

            # Add threshold lines
            fig.add_hline(
                y=self.anomaly_threshold,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=2, col=1
            )
            fig.add_hline(
                y=-self.anomaly_threshold,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=2, col=1
            )
            fig.add_hline(
                y=0,
                line_dash="solid",
                line_color="black",
                opacity=0.3,
                row=2, col=1
            )

        # Update layout
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Occurrence Count", row=1, col=1)
        fig.update_yaxes(title_text="Z-score", row=2, col=1)

        fig.update_layout(
            height=800,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def analyze(self, gbif_json: Dict, show_plot: bool = True) -> Dict:
        """
        Complete analysis pipeline for a single species with caching support.

        Args:
            gbif_json: GBIF occurrence data dictionary
            show_plot: Whether to display interactive plot

        Returns:
            Dictionary containing all analysis results
        """
        # Parse species name first for cache lookup
        species_name = gbif_json["data"]["scientific_name"]

        # Check cache first
        if self.enable_cache and self.cache:
            cache_key = self._get_cache_key(species_name)
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Anomaly detection: Cache hit for species '{species_name}'")
                # Reconstruct results from cached data
                try:
                    # Deserialize dataframes from records
                    cached_data['time_series'] = pd.DataFrame(cached_data.get('time_series', []))
                    if cached_data.get('anomaly_results'):
                        cached_data['anomaly_results'] = pd.DataFrame(cached_data['anomaly_results'])

                    # Reconstruct figure from JSON
                    if cached_data.get('figure_json'):
                        import plotly.io as pio
                        cached_data['figure'] = pio.from_json(cached_data['figure_json'])
                    else:
                        cached_data['figure'] = None

                    # Reconstruct structured results
                    if cached_data.get('structured_results'):
                        sr = cached_data['structured_results']
                        cached_data['structured_results'] = AnomalyResults(
                            species_name=sr['species_name'],
                            summary_stats=TimeSeriesStats(**sr['summary_stats']),
                            num_anomalies=sr['num_anomalies'],
                            num_declines=sr['num_declines'],
                            num_surges=sr['num_surges'],
                            episodes=[AnomalyEpisode(**ep) for ep in sr['episodes']]
                        )

                    return cached_data
                except Exception as e:
                    logger.warning(f"Anomaly detection: Failed to deserialize cached data: {e}, running fresh analysis")

        logger.info(f"Anomaly detection: Cache miss for species '{species_name}', running analysis")
        logger.info("="*80)
        logger.info("GBIF OCCURRENCE ANOMALY DETECTION")
        logger.info("="*80)

        # Parse data
        logger.info("1. Parsing GBIF data...")
        species_name, df = self.parse_gbif_json(gbif_json)
        logger.info(f"   Species: {species_name}")
        logger.info(f"   Time series length: {len(df)} years")

        # Summary statistics
        logger.info("2. Computing summary statistics...")
        stats = self.compute_summary_stats(df, species_name)
        logger.info(f"   Year range: {stats.year_range}")
        logger.info(f"   Total occurrences: {stats.total_occurrences}")
        logger.info(f"   Mean annual count: {stats.mean_count:.2f}")
        logger.info(f"   Trend: {stats.trend_direction} (slope: {stats.trend_slope:.2f})")

        # Rolling backtest
        logger.info("3. Running rolling backtest with pretrained model...")
        backtest_df = self.rolling_backtest(df)

        # Anomaly detection
        logger.info("4. Detecting anomalies...")
        if len(backtest_df) > 0:
            anomaly_df = self.compute_anomaly_scores(backtest_df)

            # Episode detection
            logger.debug("5. Grouping anomalies into episodes...")
            episodes = self.detect_episodes(anomaly_df)

            # Log detailed results at debug level
            logger.debug("="*80)
            logger.debug("RESULTS")
            logger.debug("="*80)

            # Log anomaly table at debug level
            logger.debug("Detailed Forecast Results:")
            logger.debug("\n" + anomaly_df[[
                'year', 'actual_count', 'predicted_log', 'residual',
                'z_score', 'anomaly_flag', 'anomaly_type'
            ]].to_string(index=False))

            # Log episodes at debug level
            if episodes:
                logger.debug("-"*80)
                logger.debug("DETECTED ANOMALY EPISODES")
                logger.debug("-"*80)
                for i, ep in enumerate(episodes, 1):
                    year_span = f"{ep.start_year}" if ep.start_year == ep.end_year else f"{ep.start_year}-{ep.end_year}"
                    logger.debug(f"Episode {i}:")
                    logger.debug(f"  Years: {year_span}")
                    logger.debug(f"  Type: {ep.type.upper()}")
                    logger.debug(f"  Duration: {ep.num_years} year(s)")
                    logger.debug(f"  Max |Z-score|: {ep.max_abs_z:.2f}")
                    logger.debug(f"  Mean residual: {ep.mean_residual:.3f}")

                    interpretation = (
                        "Observation counts significantly LOWER than expected"
                        if ep.type == 'decline'
                        else "Observation counts significantly HIGHER than expected"
                    )
                    logger.debug(f"  Interpretation: {interpretation}")
            else:
                logger.debug("No anomaly episodes detected.")

            # Visualization
            logger.debug("6. Creating visualization...")
            fig = self.visualize_results(df, anomaly_df, species_name, episodes)

            if show_plot:
                fig.show()

            # Create structured results
            results_obj = AnomalyResults(
                species_name=species_name,
                summary_stats=stats,
                num_anomalies=int(anomaly_df['anomaly_flag'].sum()),
                num_declines=int((anomaly_df['anomaly_type'] == 'decline').sum()),
                num_surges=int((anomaly_df['anomaly_type'] == 'surge').sum()),
                episodes=episodes
            )

        else:
            logger.warning("Insufficient data for backtesting.")
            anomaly_df = pd.DataFrame()
            episodes = []
            fig = None
            results_obj = AnomalyResults(
                species_name=species_name,
                summary_stats=stats,
                num_anomalies=0,
                num_declines=0,
                num_surges=0,
                episodes=[]
            )

        # Limitations and caveats (log at debug level)
        logger.debug("="*80)
        logger.debug("IMPORTANT LIMITATIONS")
        logger.debug("="*80)
        logger.debug("""
1. Sampling artifacts: Anomalies may reflect changes in data collection effort,
   taxonomy updates, or database ingestion patterns rather than true ecological changes.

2. Yearly resolution: The model operates at annual granularity and cannot detect
   within-year seasonal or sub-annual patterns.

3. Context dependency: Forecasts depend on having sufficient historical context.
   Early years in the time series have less reliable predictions.

4. Statistical vs. biological significance: A statistically significant anomaly
   does not automatically indicate biological importance. Domain expertise is
   required for interpretation.

5. Spatial aggregation: This analysis ignores spatial distribution. Geographic
   shifts in occurrence could appear as temporal anomalies.
        """)

        # Prepare results
        results = {
            'species_name': species_name,
            'time_series': df,
            'summary_stats': stats,
            'backtest_results': backtest_df,
            'anomaly_results': anomaly_df if len(backtest_df) > 0 else None,
            'episodes': episodes,
            'figure': fig,
            'structured_results': results_obj
        }

        # Cache results if enabled
        if self.enable_cache and self.cache:
            cache_key = self._get_cache_key(species_name)
            try:
                # Prepare cacheable version (serialize complex objects)
                cache_data = {
                    'species_name': species_name,
                    'time_series': df.to_dict('records') if df is not None else [],
                    'summary_stats': stats.model_dump() if stats else {},
                    'backtest_results': backtest_df.to_dict('records') if backtest_df is not None else [],
                    'anomaly_results': anomaly_df.to_dict('records') if anomaly_df is not None and len(anomaly_df) > 0 else None,
                    'episodes': [ep.model_dump() for ep in episodes] if episodes else [],
                    'figure_json': fig.to_json() if fig is not None else None,
                    'structured_results': results_obj.model_dump() if results_obj else None
                }
                self.cache.set(cache_key, cache_data)
                logger.debug(f"Anomaly detection: Cached results for species '{species_name}'")
            except Exception as e:
                logger.warning(f"Anomaly detection: Failed to cache results: {e}")

        return results


def main():
    """Example usage with provided Gorilla beringei data."""
    # Sample GBIF JSON
    gbif_json = {
        "_cached_at": "2025-12-11T23:56:56.181382",
        "data": {
            "scientific_name": "Gorilla beringei Matschie, 1903",
            "occurrence_count": 1180,
            "temporal_distribution": {
                "2014": 20, "2013": 27, "2012": 54, "2011": 18, "2010": 3,
                "2009": 20, "2008": 21, "2007": 4, "2006": 5, "2005": 1,
                "2004": 1, "2003": 5, "2002": 1, "2001": 3, "2000": 3,
                "1998": 1, "1996": 1, "1992": 1, "1991": 27, "1990": 2,
                "1989": 1, "1988": 1, "1986": 3, "1985": 1, "1980": 2,
                "1973": 1, "1969": 3, "1968": 1, "1959": 2, "1957": 4,
                "1954": 4, "1949": 2, "1945": 2, "1942": 2, "1938": 1,
                "1937": 12, "1936": 1, "1932": 1, "1931": 1, "1927": 4,
                "1925": 4, "1924": 2, "1921": 4, "1909": 1, "1908": 1,
                "1907": 4, "1902": 2, "2025": 80, "2024": 120, "2023": 99,
                "2022": 34, "2021": 23, "2020": 26, "2019": 55, "2018": 45,
                "2017": 37, "2016": 37, "2015": 27
            },
            "spatial_distribution": []
        }
    }

    # Initialize detector with pretrained model
    # Use "tiny" for speed; switch to "small" or "base" for better accuracy
    # device=None enables auto-detection of GPU
    detector = GBIFAnomalyDetector(
        model_size="tiny",
        context_length=15,
        forecast_horizon=3,
        anomaly_threshold=2.0,
        device=None  # Auto-detect GPU
    )

    # Run complete analysis
    results = detector.analyze(gbif_json, show_plot=True)

    # Results are returned as dictionary for programmatic access
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("Results dictionary contains:")
    logger.info(f"  - species_name: {results['species_name']}")
    logger.info(f"  - time_series: DataFrame with {len(results['time_series'])} years")
    logger.info(f"  - summary_stats: TimeSeriesStats object")
    if results['anomaly_results'] is not None:
        logger.info(f"  - anomaly_results: DataFrame with {len(results['anomaly_results'])} forecasted points")
        logger.info(f"  - episodes: {len(results['episodes'])} detected episodes")
    logger.info(f"  - figure: Plotly figure object")
    logger.info(f"  - structured_results: AnomalyResults Pydantic model")

    # Save figure to HTML
    if results['figure'] is not None:
        output_file = "gbif_anomaly_detection.html"
        results['figure'].write_html(output_file)
        logger.info(f"Visualization saved to: {output_file}")


if __name__ == "__main__":
    main()
