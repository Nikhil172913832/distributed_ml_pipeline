"""
Model export utilities for production deployment.

Supports ONNX export for optimized inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

from deep_models import DeepClassifier, Autoencoder


def export_to_onnx(
    model: torch.nn.Module,
    input_dim: int,
    output_path: Path,
    opset_version: int = 14
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        input_dim: Number of input features
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    
    Returns:
        True if export successful
    """
    try:
        model.eval()
        
        dummy_input = torch.randn(1, input_dim)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def verify_onnx_model(onnx_path: Path, input_dim: int) -> bool:
    """
    Verify ONNX model can be loaded and run.
    """
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        
        dummy_input = np.random.randn(1, input_dim).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        outputs = session.run(None, {input_name: dummy_input})
        
        logger.info(f"ONNX model verified: {onnx_path}")
        logger.info(f"Output shape: {outputs[0].shape}")
        return True
        
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return False
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


class ONNXInferenceEngine:
    """
    Optimized inference using ONNX runtime.
    Faster than PyTorch for CPU inference.
    """
    
    def __init__(self, onnx_path: Path):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx_path))
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Loaded ONNX model from {onnx_path}")
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        X = X.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: X})
        return outputs[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        logits = self.predict(X)
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
