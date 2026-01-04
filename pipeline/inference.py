"""
Model Inference Service
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import signal
from uuid import UUID

import pandas as pd
import numpy as np
import joblib
from scipy import stats
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dotenv import load_dotenv

from database import (
    DatabaseManager, PreprocessedDataRepository, ModelRegistryRepository,
    PredictionRepository, ModelPerformanceRepository, DataDriftRepository,
    RetrainingTriggerRepository, AuditLogRepository
)

load_dotenv()

CONFIG = {
    'inference': {
        'batch_size': int(os.getenv('INFERENCE_BATCH_SIZE', 100)),
        'poll_interval_seconds': int(os.getenv('INFERENCE_POLL_INTERVAL', 5)),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', 0.7)),
    },
    'performance': {
        'monitoring_window_hours': int(os.getenv('PERFORMANCE_WINDOW_HOURS', 1)),
        'accuracy_threshold': float(os.getenv('ACCURACY_THRESHOLD', 0.85)),
        'f1_threshold': float(os.getenv('F1_THRESHOLD', 0.80)),
        'degradation_tolerance': float(os.getenv('DEGRADATION_TOLERANCE', 0.05)),
    },
    'drift_detection': {
        'enabled': os.getenv('DRIFT_DETECTION_ENABLED', 'true').lower() == 'true',
        'check_interval_hours': int(os.getenv('DRIFT_CHECK_INTERVAL_HOURS', 6)),
        'ks_test_threshold': float(os.getenv('KS_TEST_THRESHOLD', 0.05)),
        'drift_score_threshold': float(os.getenv('DRIFT_SCORE_THRESHOLD', 0.3)),
        'baseline_days': int(os.getenv('BASELINE_DAYS', 7)),
    },
    'monitoring': {
        'prometheus_port': int(os.getenv('INFERENCE_PROMETHEUS_PORT', 8002)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    }
}

# ==========================================
# PROMETHEUS METRICS
# ==========================================
metrics = {
    'predictions_made': Counter(
        'secom_predictions_made_total',
        'Total predictions made'
    ),
    'inference_duration': Histogram(
        'secom_inference_duration_seconds',
        'Time to make predictions'
    ),
    'model_accuracy': Gauge(
        'secom_model_accuracy',
        'Current model accuracy'
    ),
    'model_f1_score': Gauge(
        'secom_model_f1_score',
        'Current model F1 score'
    ),
    'drift_detected': Counter(
        'secom_drift_detected_total',
        'Total drift detections',
        ['drift_type']
    ),
    'retraining_triggered': Counter(
        'secom_retraining_triggered_total',
        'Total retraining triggers',
        ['trigger_type']
    ),
    'low_confidence_predictions': Counter(
        'secom_low_confidence_predictions_total',
        'Predictions with low confidence'
    ),
    'active_predictions': Gauge(
        'secom_active_predictions',
        'Currently processing predictions'
    )
}

# ==========================================
# LOGGING SETUP
# ==========================================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=CONFIG['monitoring']['log_level']
)
logger.add(
    "logs/inference_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    level="DEBUG"
)


class ModelInferenceEngine:
    """Handles model loading and inference"""
    
    def __init__(self, model_path: str, preprocessing_path: Optional[str] = None):
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path
        self.model = None
        self.preprocessing_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            
            if self.preprocessing_path:
                logger.info(f"Loading preprocessing pipeline from {self.preprocessing_path}")
                self.preprocessing_pipeline = joblib.load(self.preprocessing_path)
                logger.info("Preprocessing pipeline loaded")
                
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_batch(self, preprocessed_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of preprocessed data
        
        Args:
            preprocessed_data: DataFrame with preprocessed features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            start_time = time.time()
            
            # Get predictions
            predictions = self.model.predict(preprocessed_data)
            
            # Get probability estimates if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(preprocessed_data)
            else:
                # For models without predict_proba, use decision function or dummy probabilities
                if hasattr(self.model, 'decision_function'):
                    decision_scores = self.model.decision_function(preprocessed_data)
                    # Convert to pseudo-probabilities using sigmoid
                    probabilities = 1 / (1 + np.exp(-decision_scores))
                    probabilities = np.column_stack([1 - probabilities, probabilities])
                else:
                    # Dummy probabilities (1.0 for predicted class, 0.0 for other)
                    probabilities = np.zeros((len(predictions), 2))
                    probabilities[predictions == -1, 0] = 1.0
                    probabilities[predictions == 1, 1] = 1.0
            
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Batch inference completed in {duration_ms:.2f}ms")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise


class DriftDetector:
    """Detects data drift using statistical tests"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.baseline_data = None
        self.baseline_stats = {}
    
    def set_baseline(self, baseline_data: pd.DataFrame):
        """Set baseline distribution for drift detection"""
        self.baseline_data = baseline_data
        self.baseline_stats = {
            'mean': baseline_data.mean(),
            'std': baseline_data.std(),
            'min': baseline_data.min(),
            'max': baseline_data.max()
        }
        logger.info(f"Baseline set with {len(baseline_data)} samples, {len(baseline_data.columns)} features")
    
    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature_name: str
    ) -> Tuple[bool, float, Dict]:
        """
        Detect drift for a specific feature using KS test
        
        Returns:
            Tuple of (is_drift, drift_score, metrics)
        """
        if self.baseline_data is None or feature_name not in self.baseline_data.columns:
            return False, 0.0, {}
        
        try:
            baseline_values = self.baseline_data[feature_name].dropna()
            current_values = current_data[feature_name].dropna()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
            
            # Calculate drift score (0-1, higher = more drift)
            drift_score = min(ks_stat * 2, 1.0)  # Scale KS statistic
            
            # Drift detected if p-value < threshold
            is_drift = ks_pvalue < self.config['ks_test_threshold']
            
            metrics = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'mean_baseline': float(baseline_values.mean()),
                'mean_current': float(current_values.mean()),
                'std_baseline': float(baseline_values.std()),
                'std_current': float(current_values.std())
            }
            
            return is_drift, drift_score, metrics
            
        except Exception as e:
            logger.error(f"Error detecting drift for {feature_name}: {e}")
            return False, 0.0, {}
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Tuple[bool, float, Dict]:
        """Detect drift in prediction distribution"""
        try:
            # Chi-square test for categorical distribution
            baseline_counts = np.bincount(baseline_predictions + 1, minlength=3)[1:]  # +1 to handle -1, 1
            current_counts = np.bincount(current_predictions + 1, minlength=3)[1:]
            
            # Normalize to proportions
            baseline_prop = baseline_counts / baseline_counts.sum()
            current_prop = current_counts / current_counts.sum()
            
            # Calculate drift score as total variation distance
            drift_score = 0.5 * np.abs(baseline_prop - current_prop).sum()
            
            is_drift = drift_score > self.config['drift_score_threshold']
            
            metrics = {
                'baseline_fail_rate': float(baseline_prop[1]),
                'current_fail_rate': float(current_prop[1]),
                'drift_score': float(drift_score)
            }
            
            return is_drift, drift_score, metrics
            
        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}")
            return False, 0.0, {}


class InferenceOrchestrator:
    """Orchestrates the inference and monitoring pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        
        # Initialize database components
        self.db_manager = DatabaseManager()
        self.preprocessed_repo = PreprocessedDataRepository(self.db_manager)
        self.model_registry_repo = ModelRegistryRepository(self.db_manager)
        self.prediction_repo = PredictionRepository(self.db_manager)
        self.performance_repo = ModelPerformanceRepository(self.db_manager)
        self.drift_repo = DataDriftRepository(self.db_manager)
        self.retraining_repo = RetrainingTriggerRepository(self.db_manager)
        self.audit_repo = AuditLogRepository(self.db_manager)
        
        # Load active model
        self.current_model = None
        self.current_model_id = None
        self.inference_engine = None
        self._load_active_model()
        
        # Initialize drift detector
        self.drift_detector = DriftDetector(config['drift_detection'])
        self._initialize_baseline()
        
        # Tracking
        self.last_performance_check = datetime.utcnow()
        self.last_drift_check = datetime.utcnow()
        self.processed_batches = set()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _load_active_model(self):
        """Load the currently active model"""
        try:
            model_info = self.model_registry_repo.get_active_model()
            
            if model_info is None:
                logger.warning("No active model found! Please deploy a model first.")
                logger.warning("Using default model path from config")
                default_path = './models/best_model_logistic_regression.joblib'
                if os.path.exists(default_path):
                    self.inference_engine = ModelInferenceEngine(default_path)
                    logger.info("Loaded default model for inference")
                else:
                    raise RuntimeError("No active model available")
            else:
                model_id, model_name, model_version, model_type, model_path, preprocessing_path, deployed_at = model_info
                self.current_model_id = UUID(model_id)
                self.inference_engine = ModelInferenceEngine(model_path, preprocessing_path)
                
                logger.info(f"Loaded active model: {model_name} v{model_version}")
                logger.info(f"  Type: {model_type}")
                logger.info(f"  Deployed: {deployed_at}")
                
        except Exception as e:
            logger.error(f"Error loading active model: {e}")
            raise
    
    def _initialize_baseline(self):
        """Initialize drift detection baseline from historical data"""
        if not self.config['drift_detection']['enabled']:
            logger.info("Drift detection disabled")
            return
        
        try:
            # Query baseline data from database
            baseline_days = self.config['drift_detection']['baseline_days']
            query = f"""
            SELECT features
            FROM secom.preprocessed_data
            WHERE created_at >= NOW() - INTERVAL '{baseline_days} days'
            LIMIT 10000
            """
            
            with self.db_manager.get_connection() as conn:
                import pandas as pd
                baseline_df = pd.read_sql(query, conn)
                
                if len(baseline_df) > 0:
                    # Extract features from JSONB
                    features_list = baseline_df['features'].apply(lambda x: pd.Series(x))
                    self.drift_detector.set_baseline(features_list)
                    logger.info(f"Drift baseline initialized with {len(features_list)} samples")
                else:
                    logger.warning("No baseline data available for drift detection")
                    
        except Exception as e:
            logger.error(f"Error initializing drift baseline: {e}")
    
    def _get_pending_preprocessed_data(self, limit: int = 100) -> List[Dict]:
        """Get preprocessed data that hasn't been predicted yet"""
        query = """
        SELECT pd.id, pd.features, pd.target, rd.batch_id
        FROM secom.preprocessed_data pd
        JOIN secom.raw_data rd ON pd.raw_data_id = rd.id
        LEFT JOIN secom.predictions p ON pd.id = p.preprocessed_data_id
        WHERE p.id IS NULL
        ORDER BY pd.created_at
        LIMIT %s
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (limit,))
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'id': row[0],
                            'features': row[1],
                            'target': row[2],
                            'batch_id': row[3]
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error fetching preprocessed data: {e}")
            raise
    
    def _make_predictions(self, data_batch: List[Dict]):
        """Make predictions on a batch of data"""
        if not data_batch:
            return
        
        try:
            metrics['active_predictions'].inc()
            start_time = time.time()
            
            # Convert to DataFrame
            features_list = [item['features'] for item in data_batch]
            df = pd.DataFrame(features_list)
            
            # Make predictions
            with metrics['inference_duration'].time():
                predictions, probabilities = self.inference_engine.predict_batch(df)
            
            # Prepare prediction records
            prediction_records = []
            inference_duration_ms = (time.time() - start_time) * 1000 / len(data_batch)
            
            for idx, item in enumerate(data_batch):
                pred_class = int(predictions[idx])
                proba_pass = float(probabilities[idx][0])  # Probability of -1 (pass)
                proba_fail = float(probabilities[idx][1])  # Probability of 1 (fail)
                pred_proba = proba_pass if pred_class == -1 else proba_fail
                
                # Calculate confidence and uncertainty
                confidence = max(proba_pass, proba_fail)
                uncertainty = 1.0 - confidence
                
                # Check correctness if target is available
                actual_target = item.get('target')
                is_correct = (pred_class == actual_target) if actual_target is not None else None
                
                # Track low confidence predictions
                if confidence < self.config['inference']['confidence_threshold']:
                    metrics['low_confidence_predictions'].inc()
                
                prediction_record = {
                    'preprocessed_data_id': str(item['id']),
                    'model_id': str(self.current_model_id) if self.current_model_id else None,
                    'prediction': pred_class,
                    'prediction_probability': pred_proba,
                    'prediction_proba_pass': proba_pass,
                    'prediction_proba_fail': proba_fail,
                    'actual_target': actual_target,
                    'is_correct': is_correct,
                    'confidence_score': confidence,
                    'uncertainty_score': uncertainty,
                    'inference_duration_ms': inference_duration_ms,
                    'batch_id': item.get('batch_id')
                }
                
                prediction_records.append(prediction_record)
            
            # Store predictions
            self.prediction_repo.insert_predictions(prediction_records)
            
            metrics['predictions_made'].inc(len(prediction_records))
            
            logger.info(
                f"Made {len(prediction_records)} predictions "
                f"(avg confidence: {np.mean([p['confidence_score'] for p in prediction_records]):.3f})"
            )
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='inference',
                event_status='success',
                batch_id=data_batch[0].get('batch_id'),
                component='inference',
                message=f"Predicted {len(prediction_records)} samples",
                metadata={
                    'avg_confidence': float(np.mean([p['confidence_score'] for p in prediction_records])),
                    'low_confidence_count': sum(1 for p in prediction_records 
                                                if p['confidence_score'] < self.config['inference']['confidence_threshold'])
                },
                duration_ms=inference_duration_ms * len(prediction_records)
            )
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            logger.error(traceback.format_exc())
        finally:
            metrics['active_predictions'].dec()
    
    def _check_model_performance(self):
        """Check model performance and trigger retraining if needed"""
        if self.current_model_id is None:
            return
        
        try:
            now = datetime.utcnow()
            window_hours = self.config['performance']['monitoring_window_hours']
            window_start = now - timedelta(hours=window_hours)
            
            # Calculate performance metrics
            metric_id = self.performance_repo.calculate_performance_window(
                model_id=self.current_model_id,
                window_start=window_start,
                window_end=now,
                window_type='hourly'
            )
            
            if metric_id:
                # Get latest performance
                perf = self.performance_repo.get_latest_performance(
                    self.current_model_id, 'hourly'
                )
                
                if perf:
                    accuracy, precision, recall, f1_score, total_preds, _, _ = perf
                    
                    # Update Prometheus metrics
                    if accuracy is not None:
                        metrics['model_accuracy'].set(accuracy)
                    if f1_score is not None:
                        metrics['model_f1_score'].set(f1_score)
                    
                    logger.info(
                        f"Model performance (last {window_hours}h): "
                        f"Accuracy={accuracy:.3f}, F1={f1_score:.3f}, "
                        f"Predictions={total_preds}"
                    )
                    
                    # Check for performance degradation
                    accuracy_threshold = self.config['performance']['accuracy_threshold']
                    f1_threshold = self.config['performance']['f1_threshold']
                    
                    if accuracy and accuracy < accuracy_threshold - self.config['performance']['degradation_tolerance']:
                        logger.warning(
                            f"⚠ Performance degradation detected! "
                            f"Accuracy {accuracy:.3f} < threshold {accuracy_threshold:.3f}"
                        )
                        
                        # Trigger retraining
                        self._trigger_retraining(
                            trigger_type='performance_degradation',
                            reason=f"Accuracy dropped to {accuracy:.3f} (threshold: {accuracy_threshold:.3f})",
                            performance_data={
                                'threshold': accuracy_threshold,
                                'current_value': accuracy
                            }
                        )
                    
                    elif f1_score and f1_score < f1_threshold - self.config['performance']['degradation_tolerance']:
                        logger.warning(
                            f"⚠ Performance degradation detected! "
                            f"F1 score {f1_score:.3f} < threshold {f1_threshold:.3f}"
                        )
                        
                        # Trigger retraining
                        self._trigger_retraining(
                            trigger_type='performance_degradation',
                            reason=f"F1 score dropped to {f1_score:.3f} (threshold: {f1_threshold:.3f})",
                            performance_data={
                                'threshold': f1_threshold,
                                'current_value': f1_score
                            }
                        )
            
            self.last_performance_check = now
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _check_data_drift(self):
        """Check for data drift and trigger retraining if needed"""
        if not self.config['drift_detection']['enabled']:
            return
        
        if self.drift_detector.baseline_data is None:
            logger.warning("No baseline data for drift detection")
            return
        
        try:
            now = datetime.utcnow()
            window_hours = self.config['drift_detection']['check_interval_hours']
            window_start = now - timedelta(hours=window_hours)
            
            # Get recent preprocessed data
            query = f"""
            SELECT features
            FROM secom.preprocessed_data
            WHERE created_at >= %s AND created_at < %s
            LIMIT 1000
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (window_start, now))
                    rows = cur.fetchall()
                    
                    if len(rows) < 50:
                        logger.debug("Not enough data for drift detection")
                        return
                    
                    # Convert to DataFrame
                    current_features = pd.DataFrame([row[0] for row in rows])
                    
                    # Check drift for sample of features (to avoid too many checks)
                    sample_features = current_features.columns[:50]  # Check first 50 features
                    drift_detected_count = 0
                    max_drift_score = 0.0
                    
                    for feature in sample_features:
                        is_drift, drift_score, drift_metrics = self.drift_detector.detect_feature_drift(
                            current_features, feature
                        )
                        
                        if is_drift:
                            drift_detected_count += 1
                            max_drift_score = max(max_drift_score, drift_score)
                            
                            # Record drift metric
                            self.drift_repo.record_drift_metric(
                                window_start=window_start,
                                window_end=now,
                                drift_type='feature_drift',
                                drift_score=drift_score,
                                is_drift_detected=True,
                                feature_name=feature,
                                statistical_metrics=drift_metrics
                            )
                            
                            metrics['drift_detected'].labels(drift_type='feature').inc()
                            
                            logger.warning(
                                f"⚠ Drift detected in feature '{feature}': "
                                f"score={drift_score:.3f}, p-value={drift_metrics.get('ks_pvalue', 0):.4f}"
                            )
                    
                    # Trigger retraining if significant drift
                    drift_threshold = len(sample_features) * 0.1  # 10% of features
                    if drift_detected_count > drift_threshold:
                        logger.warning(
                            f"⚠ Significant data drift detected! "
                            f"{drift_detected_count}/{len(sample_features)} features drifted"
                        )
                        
                        self._trigger_retraining(
                            trigger_type='data_drift',
                            reason=f"Data drift in {drift_detected_count} features (max score: {max_drift_score:.3f})",
                            drift_data={
                                'threshold': self.config['drift_detection']['drift_score_threshold'],
                                'current_value': max_drift_score
                            }
                        )
            
            self.last_drift_check = now
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            logger.error(traceback.format_exc())
    
    def _trigger_retraining(
        self,
        trigger_type: str,
        reason: str,
        performance_data: Optional[Dict] = None,
        drift_data: Optional[Dict] = None
    ):
        """Trigger model retraining"""
        try:
            trigger_id = self.retraining_repo.create_trigger(
                trigger_type=trigger_type,
                trigger_reason=reason,
                model_id=self.current_model_id,
                performance_data=performance_data,
                drift_data=drift_data
            )
            
            metrics['retraining_triggered'].labels(trigger_type=trigger_type).inc()
            
            logger.warning(f"🔄 Retraining triggered (ID: {trigger_id}): {reason}")
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='retraining_triggered',
                event_status='pending',
                component='inference',
                message=f"Retraining triggered: {reason}",
                metadata={
                    'trigger_type': trigger_type,
                    'trigger_id': str(trigger_id),
                    'performance_data': performance_data,
                    'drift_data': drift_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("SECOM Model Inference Service Started")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Batch size: {self.config['inference']['batch_size']}")
        logger.info(f"  Poll interval: {self.config['inference']['poll_interval_seconds']}s")
        logger.info(f"  Drift detection: {self.config['drift_detection']['enabled']}")
        logger.info(f"  Performance monitoring: {self.config['performance']['monitoring_window_hours']}h window")
        logger.info("=" * 80)
        
        try:
            while self.running:
                # Get pending data
                pending_data = self._get_pending_preprocessed_data(
                    limit=self.config['inference']['batch_size']
                )
                
                if pending_data:
                    logger.debug(f"Found {len(pending_data)} samples for inference")
                    self._make_predictions(pending_data)
                
                # Periodic checks
                now = datetime.utcnow()
                
                # Check model performance
                if (now - self.last_performance_check).total_seconds() >= 3600:  # Every hour
                    self._check_model_performance()
                
                # Check data drift
                if (now - self.last_drift_check).total_seconds() >= self.config['drift_detection']['check_interval_hours'] * 3600:
                    self._check_data_drift()
                
                # Sleep before next poll
                time.sleep(self.config['inference']['poll_interval_seconds'])
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down inference service...")
        self.running = False
        self.db_manager.close_all()
        
        logger.info("=" * 80)
        logger.info("Inference service shutdown complete")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    try:
        # Start Prometheus metrics server
        logger.info(f"Starting Prometheus metrics server on port {CONFIG['monitoring']['prometheus_port']}")
        start_http_server(CONFIG['monitoring']['prometheus_port'])
        
        # Create and run orchestrator
        orchestrator = InferenceOrchestrator(CONFIG)
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
