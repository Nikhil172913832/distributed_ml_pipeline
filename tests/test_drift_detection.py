"""
Data drift detection and simulation tests.

Tests drift detection algorithms and simulates various drift scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification


class DriftDetector:
    """
    Statistical drift detection using various methods.
    """
    
    @staticmethod
    def kolmogorov_smirnov_test(
        reference: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05
    ) -> tuple[bool, float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution shift.
        
        Args:
            reference: Reference distribution (baseline)
            current: Current distribution
            alpha: Significance level
            
        Returns:
            (is_drift_detected, statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        is_drift = p_value < alpha
        
        return is_drift, statistic, p_value
    
    @staticmethod
    def chi_squared_test(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
        alpha: float = 0.05
    ) -> tuple[bool, float, float]:
        """
        Perform chi-squared test for categorical drift.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for continuous data
            alpha: Significance level
            
        Returns:
            (is_drift_detected, statistic, p_value)
        """
        # Bin continuous data with same bins for both
        all_data = np.concatenate([reference, current])
        _, bin_edges = np.histogram(all_data, bins=bins)
        
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to same total
        ref_hist = ref_hist / len(reference) * len(current)
        
        # Chi-squared test
        statistic, p_value = stats.chisquare(curr_hist, ref_hist)
        is_drift = p_value < alpha
        
        return is_drift, statistic, p_value
    
    @staticmethod
    def population_stability_index(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI > 0.25: significant shift
        PSI 0.1-0.25: moderate shift
        PSI < 0.1: no significant shift
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins based on reference
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_prop = ref_hist / len(reference)
        curr_prop = curr_hist / len(current)
        
        # Avoid log(0)
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
        
        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        
        return psi
    
    @staticmethod
    def detect_multivariate_drift(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        method: str = 'ks',
        alpha: float = 0.05
    ) -> dict:
        """
        Detect drift across multiple features.
        
        Args:
            reference: Reference DataFrame
            current: Current DataFrame
            method: Test method ('ks' or 'chi2')
            alpha: Significance level
            
        Returns:
            Dictionary with drift results per feature
        """
        results = {}
        
        for column in reference.columns:
            if pd.api.types.is_numeric_dtype(reference[column]):
                if method == 'ks':
                    is_drift, stat, p_val = DriftDetector.kolmogorov_smirnov_test(
                        reference[column].dropna().values,
                        current[column].dropna().values,
                        alpha
                    )
                elif method == 'chi2':
                    is_drift, stat, p_val = DriftDetector.chi_squared_test(
                        reference[column].dropna().values,
                        current[column].dropna().values,
                        alpha=alpha
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                psi = DriftDetector.population_stability_index(
                    reference[column].dropna().values,
                    current[column].dropna().values
                )
                
                results[column] = {
                    'drift_detected': is_drift,
                    'statistic': stat,
                    'p_value': p_val,
                    'psi': psi,
                    'method': method
                }
        
        return results


class DriftSimulator:
    """
    Simulate various drift scenarios for testing.
    """
    
    @staticmethod
    def gradual_mean_shift(
        data: np.ndarray,
        shift_amount: float = 0.5,
        shift_rate: float = 0.1
    ) -> np.ndarray:
        """
        Simulate gradual mean shift (covariate drift).
        
        Args:
            data: Original data
            shift_amount: Total shift amount in standard deviations
            shift_rate: Rate of shift per sample
            
        Returns:
            Data with gradual mean shift
        """
        n_samples = len(data)
        shifts = np.linspace(0, shift_amount, n_samples)
        std = np.std(data)
        
        return data + shifts * std
    
    @staticmethod
    def sudden_distribution_change(
        data: np.ndarray,
        change_point: int,
        scale_factor: float = 1.5
    ) -> np.ndarray:
        """
        Simulate sudden distribution change (concept drift).
        
        Args:
            data: Original data
            change_point: Index where drift occurs
            scale_factor: Scaling factor for distribution
            
        Returns:
            Data with sudden distribution change
        """
        drifted = data.copy()
        drifted[change_point:] *= scale_factor
        
        return drifted
    
    @staticmethod
    def add_noise(
        data: np.ndarray,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Add Gaussian noise to simulate measurement drift.
        
        Args:
            data: Original data
            noise_level: Standard deviation of noise as fraction of data std
            
        Returns:
            Noisy data
        """
        std = np.std(data)
        noise = np.random.normal(0, noise_level * std, len(data))
        
        return data + noise
    
    @staticmethod
    def seasonal_pattern(
        n_samples: int,
        period: int = 100,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate seasonal drift pattern.
        
        Args:
            n_samples: Number of samples
            period: Period of seasonality
            amplitude: Amplitude of seasonal component
            
        Returns:
            Seasonal pattern array
        """
        t = np.arange(n_samples)
        return amplitude * np.sin(2 * np.pi * t / period)


class TestDriftDetection:
    """Test drift detection methods"""
    
    def test_no_drift(self):
        """Test with identical distributions (no drift)"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        is_drift, stat, p_val = DriftDetector.kolmogorov_smirnov_test(
            reference, current, alpha=0.05
        )
        
        # Should not detect drift
        assert not is_drift
        assert p_val > 0.05
    
    def test_mean_shift_detection(self):
        """Test detection of mean shift"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(1.0, 1, 1000)  # Mean shifted by 1
        
        is_drift, stat, p_val = DriftDetector.kolmogorov_smirnov_test(
            reference, current, alpha=0.05
        )
        
        # Should detect drift
        assert is_drift
        assert p_val < 0.05
    
    def test_variance_change_detection(self):
        """Test detection of variance change"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 3, 1000)  # Variance increased
        
        is_drift, stat, p_val = DriftDetector.kolmogorov_smirnov_test(
            reference, current, alpha=0.05
        )
        
        # Should detect drift
        assert is_drift
    
    def test_distribution_shape_change(self):
        """Test detection of distribution shape change"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.exponential(1, 1000)  # Different distribution
        
        is_drift, stat, p_val = DriftDetector.kolmogorov_smirnov_test(
            reference, current, alpha=0.05
        )
        
        # Should strongly detect drift
        assert is_drift
        assert p_val < 0.001
    
    def test_psi_calculation(self):
        """Test Population Stability Index calculation"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        
        # No drift
        current_no_drift = np.random.normal(0, 1, 1000)
        psi_no_drift = DriftDetector.population_stability_index(reference, current_no_drift)
        
        # Moderate drift
        current_moderate = np.random.normal(0.5, 1, 1000)
        psi_moderate = DriftDetector.population_stability_index(reference, current_moderate)
        
        # Significant drift
        current_significant = np.random.normal(2, 1, 1000)
        psi_significant = DriftDetector.population_stability_index(reference, current_significant)
        
        # PSI should increase with drift magnitude
        assert psi_no_drift < 0.15  # No significant shift
        assert psi_moderate > 0.1  # Some shift
        assert psi_significant > psi_moderate  # More shift
    
    def test_multivariate_drift_detection(self):
        """Test drift detection across multiple features"""
        np.random.seed(42)
        
        # Create reference data
        reference = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.exponential(1, 1000)
        })
        
        # Create current data with drift in feature_2
        current = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),  # No drift
            'feature_2': np.random.normal(7, 2, 1000),  # Mean shift
            'feature_3': np.random.exponential(1, 1000)  # No drift
        })
        
        results = DriftDetector.detect_multivariate_drift(
            reference, current, method='ks'
        )
        
        # Should detect drift only in feature_2
        assert not results['feature_1']['drift_detected']
        assert results['feature_2']['drift_detected']
        # feature_3 might or might not detect drift due to randomness


class TestDriftSimulation:
    """Test drift simulation methods"""
    
    def test_gradual_shift(self):
        """Test gradual mean shift simulation"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        drifted = DriftSimulator.gradual_mean_shift(data, shift_amount=1.0)
        
        # Mean should increase gradually
        first_half_mean = np.mean(drifted[:500])
        second_half_mean = np.mean(drifted[500:])
        
        assert second_half_mean > first_half_mean
    
    def test_sudden_change(self):
        """Test sudden distribution change"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        drifted = DriftSimulator.sudden_distribution_change(
            data, change_point=500, scale_factor=2.0
        )
        
        # Variance should be different before/after change point
        var_before = np.var(drifted[:500])
        var_after = np.var(drifted[500:])
        
        assert var_after > var_before
    
    def test_noise_addition(self):
        """Test noise addition"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        
        noisy = DriftSimulator.add_noise(data, noise_level=0.5)
        
        # Variance should increase
        assert np.var(noisy) > np.var(data)
    
    def test_seasonal_pattern(self):
        """Test seasonal pattern generation"""
        seasonal = DriftSimulator.seasonal_pattern(
            n_samples=1000,
            period=100,
            amplitude=1.0
        )
        
        # Should be periodic
        # Check that pattern repeats
        first_period = seasonal[:100]
        second_period = seasonal[100:200]
        
        # Should be similar (allowing for small numerical differences)
        correlation = np.corrcoef(first_period, second_period)[0, 1]
        assert correlation > 0.95


class TestDriftMonitoring:
    """Test drift monitoring scenarios"""
    
    def test_monitoring_window(self):
        """Test sliding window drift detection"""
        np.random.seed(42)
        
        # Generate data with drift at t=500
        baseline = np.random.normal(0, 1, 500)
        drifted = np.random.normal(1, 1, 500)
        data_stream = np.concatenate([baseline, drifted])
        
        # Monitor with sliding window
        window_size = 100
        alerts = []
        
        for i in range(window_size, len(data_stream), window_size):
            reference_window = baseline[-window_size:]
            current_window = data_stream[i-window_size:i]
            
            is_drift, _, _ = DriftDetector.kolmogorov_smirnov_test(
                reference_window, current_window
            )
            
            if is_drift:
                alerts.append(i)
        
        # Should detect drift in second half
        assert len(alerts) > 0
        assert all(alert >= 500 for alert in alerts)
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold for drift detection"""
        np.random.seed(42)
        
        reference = np.random.normal(0, 1, 1000)
        
        # Test different alpha levels
        alphas = [0.01, 0.05, 0.1]
        detection_rates = []
        
        for alpha in alphas:
            detections = 0
            for _ in range(100):
                current = np.random.normal(0.5, 1, 1000)
                is_drift, _, _ = DriftDetector.kolmogorov_smirnov_test(
                    reference, current, alpha=alpha
                )
                if is_drift:
                    detections += 1
            
            detection_rates.append(detections / 100)
        
        # More lenient alpha should detect more drift
        assert detection_rates[2] >= detection_rates[1] >= detection_rates[0]
    
    @pytest.mark.parametrize("drift_type,detector_method", [
        ("mean_shift", "ks"),
        ("variance_change", "ks"),
        ("distribution_change", "chi2"),
    ])
    def test_drift_type_detection(self, drift_type, detector_method):
        """Test detection of different drift types"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        
        if drift_type == "mean_shift":
            current = np.random.normal(1, 1, 1000)
        elif drift_type == "variance_change":
            current = np.random.normal(0, 2, 1000)
        elif drift_type == "distribution_change":
            current = np.random.exponential(1, 1000)
        
        if detector_method == "ks":
            is_drift, _, _ = DriftDetector.kolmogorov_smirnov_test(reference, current)
        elif detector_method == "chi2":
            is_drift, _, _ = DriftDetector.chi_squared_test(reference, current)
        
        # Should detect all drift types
        assert is_drift


@pytest.fixture
def drift_detector():
    """Fixture providing drift detector instance"""
    return DriftDetector()


@pytest.fixture
def drift_simulator():
    """Fixture providing drift simulator instance"""
    return DriftSimulator()
