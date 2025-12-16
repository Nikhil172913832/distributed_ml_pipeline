"""
Shadow Mode Deployment Infrastructure.

Enables A/B testing by running new models in shadow mode alongside
production models without affecting user-facing predictions.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment modes for models."""
    PRODUCTION = "production"
    SHADOW = "shadow"
    CANARY = "canary"
    OFFLINE = "offline"


@dataclass
class ModelDeployment:
    """Model deployment configuration."""
    model_id: str
    model_version: str
    mode: DeploymentMode
    traffic_percentage: float = 0.0  # For canary deployments
    deployed_at: datetime = None
    
    def __post_init__(self):
        if self.deployed_at is None:
            self.deployed_at = datetime.utcnow()


class ShadowModeManager:
    """
    Manages shadow mode deployments for A/B testing.
    
    Shadow mode allows running new models alongside production models
    to compare performance without affecting user-facing predictions.
    
    Usage:
        manager = ShadowModeManager()
        
        # Deploy model in shadow mode
        manager.deploy_shadow_model(model_id="model_v2", model_version="2.0.0")
        
        # Make predictions with both models
        results = manager.predict_with_shadow(X_sample)
        
        # Compare performance
        comparison = manager.compare_models()
    """
    
    def __init__(self, database_manager=None, metrics_tracker=None):
        """
        Initialize shadow mode manager.
        
        Args:
            database_manager: Database manager for storing results
            metrics_tracker: Metrics tracker for logging
        """
        self.db = database_manager
        self.metrics = metrics_tracker
        self.deployments: Dict[str, ModelDeployment] = {}
        self.shadow_results: List[Dict[str, Any]] = []
    
    def deploy_shadow_model(
        self,
        model_id: str,
        model_version: str,
        model_path: str
    ) -> bool:
        """
        Deploy a model in shadow mode.
        
        Args:
            model_id: Unique model identifier
            model_version: Model version
            model_path: Path to model file
            
        Returns:
            True if deployment successful
        """
        try:
            deployment = ModelDeployment(
                model_id=model_id,
                model_version=model_version,
                mode=DeploymentMode.SHADOW
            )
            
            self.deployments[model_id] = deployment
            
            logger.info(f"Deployed model {model_id} v{model_version} in shadow mode")
            
            # Log to database if available
            if self.db:
                self._log_deployment(deployment)
            
            return True
        except Exception as e:
            logger.error(f"Failed to deploy shadow model: {e}")
            return False
    
    def predict_with_shadow(
        self,
        X: Any,
        production_model: Any,
        shadow_model: Any
    ) -> Dict[str, Any]:
        """
        Make predictions with both production and shadow models.
        
        Args:
            X: Input features
            production_model: Production model
            shadow_model: Shadow model
            
        Returns:
            Dictionary with both predictions and metadata
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "production": {},
            "shadow": {},
            "comparison": {}
        }
        
        try:
            # Production prediction (user-facing)
            prod_pred = production_model.predict(X)
            prod_proba = production_model.predict_proba(X) if hasattr(production_model, 'predict_proba') else None
            
            result["production"] = {
                "prediction": int(prod_pred[0]),
                "probability": float(prod_proba[0][1]) if prod_proba is not None else None
            }
            
            # Shadow prediction (not user-facing)
            shadow_pred = shadow_model.predict(X)
            shadow_proba = shadow_model.predict_proba(X) if hasattr(shadow_model, 'predict_proba') else None
            
            result["shadow"] = {
                "prediction": int(shadow_pred[0]),
                "probability": float(shadow_proba[0][1]) if shadow_proba is not None else None
            }
            
            # Compare predictions
            result["comparison"] = {
                "predictions_match": prod_pred[0] == shadow_pred[0],
                "probability_diff": abs(
                    (prod_proba[0][1] if prod_proba is not None else 0) -
                    (shadow_proba[0][1] if shadow_proba is not None else 0)
                ) if prod_proba is not None and shadow_proba is not None else None
            }
            
            # Store result for analysis
            self.shadow_results.append(result)
            
            # Log to metrics if available
            if self.metrics:
                self._log_shadow_metrics(result)
            
        except Exception as e:
            logger.error(f"Shadow prediction failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def compare_models(
        self,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Compare production and shadow model performance.
        
        Args:
            window_hours: Time window for comparison
            
        Returns:
            Comparison metrics
        """
        if not self.shadow_results:
            return {"error": "No shadow results available"}
        
        # Filter results by time window
        cutoff = datetime.utcnow().timestamp() - (window_hours * 3600)
        recent_results = [
            r for r in self.shadow_results
            if datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff
        ]
        
        if not recent_results:
            return {"error": "No recent shadow results"}
        
        # Calculate agreement metrics
        total = len(recent_results)
        matches = sum(1 for r in recent_results if r["comparison"]["predictions_match"])
        
        prob_diffs = [
            r["comparison"]["probability_diff"]
            for r in recent_results
            if r["comparison"]["probability_diff"] is not None
        ]
        
        comparison = {
            "total_predictions": total,
            "agreement_rate": matches / total if total > 0 else 0,
            "disagreement_count": total - matches,
            "avg_probability_diff": sum(prob_diffs) / len(prob_diffs) if prob_diffs else None,
            "max_probability_diff": max(prob_diffs) if prob_diffs else None,
            "window_hours": window_hours
        }
        
        return comparison
    
    def promote_shadow_to_production(
        self,
        model_id: str,
        approval_threshold: float = 0.95
    ) -> bool:
        """
        Promote shadow model to production if it meets criteria.
        
        Args:
            model_id: Model to promote
            approval_threshold: Minimum agreement rate required
            
        Returns:
            True if promotion successful
        """
        comparison = self.compare_models()
        
        if "error" in comparison:
            logger.warning(f"Cannot promote: {comparison['error']}")
            return False
        
        agreement_rate = comparison["agreement_rate"]
        
        if agreement_rate < approval_threshold:
            logger.warning(
                f"Model {model_id} agreement rate {agreement_rate:.2%} "
                f"below threshold {approval_threshold:.2%}"
            )
            return False
        
        # Update deployment mode
        if model_id in self.deployments:
            self.deployments[model_id].mode = DeploymentMode.PRODUCTION
            logger.info(f"Promoted model {model_id} to production")
            
            if self.db:
                self._log_promotion(model_id, comparison)
            
            return True
        
        return False
    
    def get_shadow_report(self) -> Dict[str, Any]:
        """Generate shadow mode deployment report."""
        active_shadows = [
            d for d in self.deployments.values()
            if d.mode == DeploymentMode.SHADOW
        ]
        
        comparison = self.compare_models()
        
        return {
            "active_shadow_models": len(active_shadows),
            "shadow_deployments": [asdict(d) for d in active_shadows],
            "total_shadow_predictions": len(self.shadow_results),
            "performance_comparison": comparison,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _log_deployment(self, deployment: ModelDeployment):
        """Log deployment to database."""
        # Implementation depends on database schema
        pass
    
    def _log_shadow_metrics(self, result: Dict[str, Any]):
        """Log shadow prediction metrics."""
        # Implementation depends on metrics system
        pass
    
    def _log_promotion(self, model_id: str, comparison: Dict[str, Any]):
        """Log model promotion event."""
        logger.info(
            f"Model promotion: {model_id}, "
            f"agreement_rate={comparison['agreement_rate']:.2%}"
        )


# ==========================================
# CANARY DEPLOYMENT
# ==========================================

class CanaryDeployment:
    """
    Canary deployment strategy for gradual rollout.
    
    Gradually increases traffic to new model while monitoring performance.
    """
    
    def __init__(self, initial_traffic: float = 0.05):
        """
        Initialize canary deployment.
        
        Args:
            initial_traffic: Initial traffic percentage (0.0-1.0)
        """
        self.traffic_percentage = initial_traffic
        self.step_size = 0.05
        self.max_traffic = 1.0
    
    def should_use_canary(self) -> bool:
        """Determine if request should use canary model."""
        import random
        return random.random() < self.traffic_percentage
    
    def increase_traffic(self, step_size: Optional[float] = None):
        """Increase canary traffic percentage."""
        step = step_size or self.step_size
        self.traffic_percentage = min(
            self.traffic_percentage + step,
            self.max_traffic
        )
        logger.info(f"Canary traffic increased to {self.traffic_percentage:.1%}")
    
    def decrease_traffic(self, step_size: Optional[float] = None):
        """Decrease canary traffic percentage."""
        step = step_size or self.step_size
        self.traffic_percentage = max(
            self.traffic_percentage - step,
            0.0
        )
        logger.info(f"Canary traffic decreased to {self.traffic_percentage:.1%}")
    
    def rollback(self):
        """Rollback canary deployment."""
        self.traffic_percentage = 0.0
        logger.warning("Canary deployment rolled back")


if __name__ == "__main__":
    # Example usage
    manager = ShadowModeManager()
    
    # Deploy shadow model
    manager.deploy_shadow_model(
        model_id="model_v2",
        model_version="2.0.0",
        model_path="./models/model_v2.joblib"
    )
    
    # Get report
    report = manager.get_shadow_report()
    print(json.dumps(report, indent=2, default=str))
