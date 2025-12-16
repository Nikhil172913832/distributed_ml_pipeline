"""
Data Validation Module using Great Expectations.

Provides comprehensive data quality checks for the SECOM dataset
to catch data issues early and prevent model degradation.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations.checkpoint import Checkpoint
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    logging.warning("Great Expectations not installed. Data validation will be limited.")

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates SECOM manufacturing data using Great Expectations.
    
    Checks include:
    - Feature count validation (590 features expected)
    - Missing value thresholds
    - Value range validation
    - Target variable validation
    - Data type consistency
    """
    
    def __init__(self, context_root_dir: Optional[Path] = None):
        """
        Initialize data validator.
        
        Args:
            context_root_dir: Root directory for GX context (default: ./gx)
        """
        self.context_root_dir = context_root_dir or Path("./gx")
        self.context = None
        self.suite_name = "secom_data_suite"
        
        if GX_AVAILABLE:
            self._initialize_context()
        else:
            logger.warning("Running in fallback mode without Great Expectations")
    
    def _initialize_context(self):
        """Initialize Great Expectations context."""
        try:
            if self.context_root_dir.exists():
                self.context = gx.get_context(context_root_dir=str(self.context_root_dir))
            else:
                self.context = gx.get_context(mode="file", context_root_dir=str(self.context_root_dir))
                self._create_expectation_suite()
        except Exception as e:
            logger.error(f"Failed to initialize GX context: {e}")
            self.context = None
    
    def _create_expectation_suite(self):
        """Create expectation suite for SECOM data."""
        if not self.context:
            return
        
        try:
            suite = self.context.add_expectation_suite(expectation_suite_name=self.suite_name)
        except Exception:
            # Suite already exists
            suite = self.context.get_expectation_suite(expectation_suite_name=self.suite_name)
        
        # Add expectations
        expectations = [
            # Table-level expectations
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 100000}
            },
            {
                "expectation_type": "expect_table_column_count_to_equal",
                "kwargs": {"value": 591}  # 590 features + 1 target
            },
            
            # Target column expectations
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": "target", "value_set": [0, 1, -1, 1]}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "target"}
            },
        ]
        
        for exp in expectations:
            try:
                suite.add_expectation(**exp)
            except Exception as e:
                logger.debug(f"Expectation already exists or failed: {e}")
        
        try:
            self.context.add_or_update_expectation_suite(expectation_suite=suite)
        except Exception as e:
            logger.error(f"Failed to save expectation suite: {e}")
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw data from Kafka.
        
        Args:
            df: DataFrame with raw SECOM data
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Basic validations (always run)
        if df.empty:
            results["valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
        
        # Check column count
        expected_cols = 591  # 590 features + target
        if len(df.columns) != expected_cols:
            results["valid"] = False
            results["errors"].append(
                f"Expected {expected_cols} columns, got {len(df.columns)}"
            )
        
        # Check for target column
        if 'target' not in df.columns:
            results["valid"] = False
            results["errors"].append("Missing 'target' column")
        else:
            # Validate target values
            unique_targets = df['target'].unique()
            valid_targets = {-1, 0, 1}
            invalid_targets = set(unique_targets) - valid_targets
            if invalid_targets:
                results["valid"] = False
                results["errors"].append(
                    f"Invalid target values: {invalid_targets}"
                )
        
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).max()
        if missing_pct > 50:
            results["warnings"].append(
                f"High missing value percentage: {missing_pct:.2f}%"
            )
        
        # Run Great Expectations if available
        if GX_AVAILABLE and self.context:
            try:
                gx_results = self._run_gx_validation(df)
                if not gx_results["success"]:
                    results["valid"] = False
                    results["errors"].extend(gx_results["errors"])
            except Exception as e:
                logger.warning(f"GX validation failed: {e}")
                results["warnings"].append(f"Advanced validation skipped: {e}")
        
        return results
    
    def validate_preprocessed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate preprocessed data before inference.
        
        Args:
            df: DataFrame with preprocessed features
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if df.empty:
            results["valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
        
        # Check for missing values (should be none after preprocessing)
        if df.isnull().any().any():
            results["valid"] = False
            missing_cols = df.columns[df.isnull().any()].tolist()
            results["errors"].append(
                f"Preprocessed data contains missing values in columns: {missing_cols}"
            )
        
        # Check value ranges (after standardization, should be roughly -10 to 10)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min < -20 or col_max > 20:
                results["warnings"].append(
                    f"Column {col} has unusual range: [{col_min:.2f}, {col_max:.2f}]"
                )
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=['float64']).values).any():
            results["valid"] = False
            results["errors"].append("Preprocessed data contains infinite values")
        
        return results
    
    def _run_gx_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Great Expectations validation."""
        if not self.context:
            return {"success": True, "errors": []}
        
        try:
            # Create batch
            batch = self.context.sources.add_pandas("pandas_source").add_dataframe_asset(
                name="secom_data"
            ).add_batch_definition_whole_dataframe("batch_def").get_batch(batch_parameters={"dataframe": df})
            
            # Get expectation suite
            suite = self.context.get_expectation_suite(expectation_suite_name=self.suite_name)
            
            # Validate
            validation_results = batch.validate(suite)
            
            errors = []
            if not validation_results.success:
                for result in validation_results.results:
                    if not result.success:
                        errors.append(result.expectation_config.expectation_type)
            
            return {
                "success": validation_results.success,
                "errors": errors
            }
        except Exception as e:
            logger.error(f"GX validation error: {e}")
            return {"success": False, "errors": [str(e)]}
    
    def get_validation_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation summary."""
        if results["valid"]:
            return "✓ Data validation passed"
        else:
            errors = "\n".join(f"  - {e}" for e in results["errors"])
            warnings = "\n".join(f"  - {w}" for w in results["warnings"]) if results["warnings"] else ""
            
            summary = f"✗ Data validation failed:\n{errors}"
            if warnings:
                summary += f"\n\nWarnings:\n{warnings}"
            return summary


# Fallback validation functions (when GX not available)
def validate_feature_count(df: pd.DataFrame, expected: int = 590) -> bool:
    """Validate number of features."""
    return len(df.columns) - 1 == expected  # -1 for target column


def validate_target_values(df: pd.DataFrame) -> bool:
    """Validate target column values."""
    if 'target' not in df.columns:
        return False
    return df['target'].isin([-1, 0, 1]).all()


def validate_no_missing_after_preprocessing(df: pd.DataFrame) -> bool:
    """Validate no missing values in preprocessed data."""
    return not df.isnull().any().any()


def validate_value_ranges(df: pd.DataFrame, min_val: float = -20, max_val: float = 20) -> bool:
    """Validate feature values are within expected range."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    return (numeric_df.min().min() >= min_val) and (numeric_df.max().max() <= max_val)


# Quick validation function for use in pipelines
def quick_validate(df: pd.DataFrame, stage: str = "raw") -> tuple[bool, str]:
    """
    Quick validation for pipeline use.
    
    Args:
        df: DataFrame to validate
        stage: "raw" or "preprocessed"
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Empty DataFrame"
    
    if stage == "raw":
        if not validate_target_values(df):
            return False, "Invalid target values"
        if len(df.columns) != 591:
            return False, f"Expected 591 columns, got {len(df.columns)}"
    
    elif stage == "preprocessed":
        if not validate_no_missing_after_preprocessing(df):
            return False, "Missing values in preprocessed data"
        if not validate_value_ranges(df):
            return False, "Values outside expected range"
    
    return True, "OK"


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    sample_data = pd.DataFrame({
        **{f"feature_{i}": np.random.randn(100) for i in range(590)},
        "target": np.random.choice([0, 1], 100)
    })
    
    # Validate
    validator = DataValidator()
    results = validator.validate_raw_data(sample_data)
    print(validator.get_validation_summary(results))
