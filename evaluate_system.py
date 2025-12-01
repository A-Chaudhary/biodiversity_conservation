"""
Evaluation Script for Biodiversity Conservation Intelligence System

This script runs comprehensive evaluation metrics on a test set of species.
Run this to generate quantitative metrics for your evaluation section.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import numpy as np
from biodiversity_intel.workflow import run_conservation_analysis
from biodiversity_intel.analysis import EvaluationMetrics
from biodiversity_intel.config import setup_logging
import time

# Setup logging
logger = setup_logging("INFO")
eval_logger = logging.getLogger("evaluation")


class SystemEvaluator:
    """Evaluates the biodiversity conservation intelligence system."""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
        self.results = []
        
    async def evaluate_species(self, species_name: str, ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate system performance on a single species.
        
        Args:
            species_name: Scientific name of species
            ground_truth: Optional ground truth data (IUCN threats, status, etc.)
            
        Returns:
            Evaluation results dictionary
        """
        eval_logger.info(f"Evaluating species: {species_name}")
        
        result = {
            "species_name": species_name,
            "timestamp": time.time(),
            "metrics": {},
            "errors": []
        }
        
        try:
            # Measure execution time
            start_time = time.time()
            system_output = await run_conservation_analysis(species_name)
            execution_time = time.time() - start_time
            
            result["execution_time"] = execution_time
            result["system_output"] = system_output
            
            # Extract metrics if ground truth available
            if ground_truth:
                # Threat detection recall
                detected_threats = system_output.get("threats", [])
                iucn_threats = ground_truth.get("iucn_threats", [])
                
                if iucn_threats:
                    recall = self.metrics.threat_detection_recall(detected_threats, iucn_threats)
                    result["metrics"]["threat_recall"] = recall
                    
                    # Calculate precision (if we can identify false positives)
                    # This requires manual annotation or ground truth
                    true_positives = len(set(detected_threats) & set(iucn_threats))
                    false_positives = len(set(detected_threats) - set(iucn_threats))
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                    result["metrics"]["threat_precision"] = precision
                    
                    # F1 Score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    result["metrics"]["threat_f1"] = f1
                
                # Population trend alignment
                system_trend = system_output.get("population_trend", "Unknown")
                iucn_trend = ground_truth.get("population_trend", "Unknown")
                result["metrics"]["trend_match"] = (system_trend.lower() == iucn_trend.lower())
                
                # Conservation status match
                system_status = system_output.get("conservation_status", "Unknown")
                iucn_status = ground_truth.get("conservation_status", "Unknown")
                result["metrics"]["status_match"] = (system_status == iucn_status)
            
            # Data availability metrics
            result["metrics"]["has_iucn_data"] = bool(system_output.get("iucn_data"))
            result["metrics"]["has_gbif_data"] = bool(system_output.get("gbif_data"))
            result["metrics"]["has_news_data"] = bool(system_output.get("news_data"))
            result["metrics"]["confidence_score"] = system_output.get("confidence_score", 0.0)
            result["metrics"]["early_warning"] = system_output.get("early_warning", False)
            
            # Count threats detected
            result["metrics"]["threats_detected"] = len(system_output.get("threats", []))
            
            # Calculate confidence alignment score (IUCN vs GBIF trends)
            iucn_data = system_output.get("iucn_data", {})
            gbif_data = system_output.get("gbif_data", {})
            
            if iucn_data and gbif_data:
                iucn_trend = iucn_data.get("population_trend", "Unknown")
                
                # Derive GBIF trend from temporal distribution
                gbif_trend = "Unknown"
                temporal_dist = gbif_data.get("temporal_distribution", {})
                if temporal_dist:
                    # Analyze recent years (last 5 years) vs previous 5 years
                    years = sorted([int(y) for y in temporal_dist.keys() if y.isdigit()], reverse=True)
                    if len(years) >= 10:
                        recent_years = years[:5]
                        previous_years = years[5:10]
                        recent_count = sum(temporal_dist.get(str(y), 0) for y in recent_years)
                        previous_count = sum(temporal_dist.get(str(y), 0) for y in previous_years)
                        
                        if recent_count > previous_count * 1.1:
                            gbif_trend = "Increasing"
                        elif recent_count < previous_count * 0.9:
                            gbif_trend = "Decreasing"
                        else:
                            gbif_trend = "Stable"
                
                alignment_score = self.metrics.confidence_alignment_score(iucn_trend, gbif_trend)
                result["metrics"]["confidence_alignment_score"] = alignment_score
                result["metrics"]["iucn_trend"] = iucn_trend
                result["metrics"]["gbif_trend"] = gbif_trend
            
            eval_logger.info(f"Evaluation complete for {species_name}: {execution_time:.2f}s")
            
        except Exception as e:
            eval_logger.error(f"Error evaluating {species_name}: {e}", exc_info=True)
            result["errors"].append(str(e))
            result["execution_time"] = None
            
        return result
    
    async def evaluate_test_set(self, test_species: List[str], ground_truth_data: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Evaluate system on multiple species.
        
        Args:
            test_species: List of species names to evaluate
            ground_truth_data: Dictionary mapping species names to ground truth
            
        Returns:
            DataFrame with evaluation results
        """
        eval_logger.info(f"Starting evaluation on {len(test_species)} species")
        
        results = []
        for species in test_species:
            ground_truth = ground_truth_data.get(species, {}) if ground_truth_data else None
            result = await self.evaluate_species(species, ground_truth)
            results.append(result)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
        
        self.results = results
        return self._results_to_dataframe(results)
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results list to pandas DataFrame."""
        rows = []
        for result in results:
            row = {
                "species_name": result["species_name"],
                "execution_time": result.get("execution_time"),
                "threat_recall": result.get("metrics", {}).get("threat_recall"),
                "threat_precision": result.get("metrics", {}).get("threat_precision"),
                "threat_f1": result.get("metrics", {}).get("threat_f1"),
                "trend_match": result.get("metrics", {}).get("trend_match"),
                "status_match": result.get("metrics", {}).get("status_match"),
                "has_iucn_data": result.get("metrics", {}).get("has_iucn_data"),
                "has_gbif_data": result.get("metrics", {}).get("has_gbif_data"),
                "has_news_data": result.get("metrics", {}).get("has_news_data"),
                "confidence_score": result.get("metrics", {}).get("confidence_score"),
                "early_warning": result.get("metrics", {}).get("early_warning"),
                "threats_detected": result.get("metrics", {}).get("threats_detected"),
                "confidence_alignment_score": result.get("metrics", {}).get("confidence_alignment_score"),
                "iucn_trend": result.get("metrics", {}).get("iucn_trend"),
                "gbif_trend": result.get("metrics", {}).get("gbif_trend"),
                "has_errors": len(result.get("errors", [])) > 0
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results."""
        def convert_to_native(value):
            """Convert numpy/pandas types to native Python types for JSON serialization."""
            if pd.isna(value):
                return None
            if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                return None
            # Convert numpy types to Python native types
            if hasattr(value, 'item'):  # numpy scalar
                return value.item()
            if isinstance(value, (pd.Series, pd.DataFrame)):
                return value.tolist() if hasattr(value, 'tolist') else None
            # Handle numpy integer and float types
            try:
                if isinstance(value, (np.integer, np.floating)):
                    return float(value) if isinstance(value, np.floating) else int(value)
            except (NameError, AttributeError):
                # numpy not available or value is not numpy type
                pass
            return value
        
        summary = {
            "total_species": int(len(df)),
            "average_execution_time": convert_to_native(df["execution_time"].mean()) if "execution_time" in df.columns else None,
            "data_availability": {
                "iucn_coverage": convert_to_native(df["has_iucn_data"].sum() / len(df)) if "has_iucn_data" in df.columns else None,
                "gbif_coverage": convert_to_native(df["has_gbif_data"].sum() / len(df)) if "has_gbif_data" in df.columns else None,
                "news_coverage": convert_to_native(df["has_news_data"].sum() / len(df)) if "has_news_data" in df.columns else None,
            },
            "average_confidence": convert_to_native(df["confidence_score"].mean()) if "confidence_score" in df.columns else None,
            "early_warnings_detected": int(df["early_warning"].sum()) if "early_warning" in df.columns else None,
            "average_confidence_alignment": convert_to_native(df["confidence_alignment_score"].mean()) if "confidence_alignment_score" in df.columns else None,
        }
        
        # Threat detection metrics (if available)
        if "threat_recall" in df.columns:
            summary["threat_detection"] = {
                "average_recall": convert_to_native(df["threat_recall"].mean()),
                "average_precision": convert_to_native(df["threat_precision"].mean()),
                "average_f1": convert_to_native(df["threat_f1"].mean()),
            }
        
        # Accuracy metrics (if ground truth available)
        if "trend_match" in df.columns:
            summary["trend_accuracy"] = convert_to_native(df["trend_match"].sum() / len(df))
        if "status_match" in df.columns:
            summary["status_accuracy"] = convert_to_native(df["status_match"].sum() / len(df))
        
        return summary
    
    def save_results(self, df: pd.DataFrame, output_dir: Path = Path("data/outputs")):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame as CSV
        csv_path = output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        eval_logger.info(f"Saved results to {csv_path}")
        
        # Save summary statistics as JSON
        summary = self.generate_summary_statistics(df)
        json_path = output_dir / "evaluation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        eval_logger.info(f"Saved summary to {json_path}")
        
        # Save full results as JSON
        full_json_path = output_dir / "evaluation_full_results.json"
        with open(full_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        eval_logger.info(f"Saved full results to {full_json_path}")
        
        return summary


async def main():
    """Main evaluation function."""
    print("=" * 80)
    print("  Biodiversity Conservation Intelligence System - Evaluation")
    print("=" * 80 + "\n")
    
    # Test species from your config
    test_species = [
        "Panthera tigris",  # Tiger
        "Ailuropoda melanoleuca",  # Giant Panda
        "Gorilla beringei"  # Mountain Gorilla
    ]
    
    # Optional: Add ground truth data for more accurate evaluation
    # You can extract this from your existing results or IUCN API
    ground_truth_data = {
        # Example structure:
        # "Panthera tigris": {
        #     "iucn_threats": ["5_1_3", "2_2_2", ...],  # From IUCN API
        #     "population_trend": "Decreasing",
        #     "conservation_status": "EN"
        # }
    }
    
    evaluator = SystemEvaluator()
    
    print(f"Evaluating {len(test_species)} species...\n")
    df = await evaluator.evaluate_test_set(test_species, ground_truth_data)
    
    print("\n" + "=" * 80)
    print("  Evaluation Results")
    print("=" * 80 + "\n")
    
    # Display results
    print(df.to_string())
    
    # Generate and display summary
    summary = evaluator.generate_summary_statistics(df)
    print("\n" + "=" * 80)
    print("  Summary Statistics")
    print("=" * 80 + "\n")
    print(json.dumps(summary, indent=2))
    
    # Save results
    summary = evaluator.save_results(df)
    
    print("\n" + "=" * 80)
    print("  Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to data/outputs/")
    print(f"- evaluation_results.csv")
    print(f"- evaluation_summary.json")
    print(f"- evaluation_full_results.json")


if __name__ == "__main__":
    # Fix Windows console encoding
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    asyncio.run(main())

