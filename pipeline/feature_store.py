"""
Feature Store Implementation using Redis.

Provides caching and retrieval of preprocessed features to avoid
recomputation and enable feature reuse across training and inference.
"""

from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime, timedelta
import hashlib

try:
    import redis
    from redis import Redis, ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not installed. Feature store will be disabled.")

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Redis-based feature store for caching preprocessed features.
    
    Features:
    - Store and retrieve features by sample ID
    - Batch operations for efficiency
    - TTL support for automatic expiration
    - Feature versioning
    - Statistics tracking
    
    Usage:
        store = FeatureStore(host="localhost", port=6379)
        
        # Store features
        store.store_features("sample_123", {"feature_1": 0.5, "feature_2": 1.2})
        
        # Retrieve features
        features = store.get_features("sample_123")
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        enabled: bool = True,
        default_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize feature store.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            enabled: Whether feature store is enabled
            default_ttl: Default TTL in seconds (24 hours)
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.default_ttl = default_ttl
        self.client: Optional[Redis] = None
        
        if self.enabled:
            try:
                pool = ConnectionPool(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    max_connections=50
                )
                self.client = Redis(connection_pool=pool)
                self.client.ping()
                logger.info(f"Feature store connected to Redis at {host}:{port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enabled = False
        else:
            logger.info("Feature store disabled")
    
    def store_features(
        self,
        sample_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
        version: str = "v1"
    ) -> bool:
        """
        Store features for a sample.
        
        Args:
            sample_id: Unique sample identifier
            features: Dictionary of feature name -> value
            ttl: Time-to-live in seconds (None = use default)
            version: Feature version
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            key = self._make_key(sample_id, version)
            value = json.dumps(features, default=str)
            ttl = ttl or self.default_ttl
            
            self.client.setex(key, ttl, value)
            
            # Update metadata
            self._update_metadata(sample_id, version, len(features))
            
            return True
        except Exception as e:
            logger.error(f"Failed to store features for {sample_id}: {e}")
            return False
    
    def get_features(
        self,
        sample_id: str,
        version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features for a sample.
        
        Args:
            sample_id: Sample identifier
            version: Feature version
            
        Returns:
            Dictionary of features or None if not found
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            key = self._make_key(sample_id, version)
            value = self.client.get(key)
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get features for {sample_id}: {e}")
            return None
    
    def store_batch(
        self,
        samples: List[Dict[str, Any]],
        id_field: str = "sample_id",
        ttl: Optional[int] = None,
        version: str = "v1"
    ) -> int:
        """
        Store features for multiple samples in batch.
        
        Args:
            samples: List of sample dictionaries (must include id_field)
            id_field: Field name containing sample ID
            ttl: Time-to-live in seconds
            version: Feature version
            
        Returns:
            Number of samples successfully stored
        """
        if not self.enabled or not self.client:
            return 0
        
        stored_count = 0
        ttl = ttl or self.default_ttl
        
        try:
            pipe = self.client.pipeline()
            
            for sample in samples:
                if id_field not in sample:
                    continue
                
                sample_id = sample[id_field]
                features = {k: v for k, v in sample.items() if k != id_field}
                
                key = self._make_key(sample_id, version)
                value = json.dumps(features, default=str)
                
                pipe.setex(key, ttl, value)
                stored_count += 1
            
            pipe.execute()
            
            logger.info(f"Stored {stored_count} samples in feature store")
            return stored_count
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return 0
    
    def get_batch(
        self,
        sample_ids: List[str],
        version: str = "v1"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for multiple samples.
        
        Args:
            sample_ids: List of sample IDs
            version: Feature version
            
        Returns:
            Dictionary mapping sample_id -> features
        """
        if not self.enabled or not self.client:
            return {}
        
        try:
            pipe = self.client.pipeline()
            keys = [self._make_key(sid, version) for sid in sample_ids]
            
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            features_map = {}
            for sample_id, result in zip(sample_ids, results):
                if result:
                    features_map[sample_id] = json.loads(result)
            
            return features_map
        except Exception as e:
            logger.error(f"Failed to get batch: {e}")
            return {}
    
    def delete_features(self, sample_id: str, version: str = "v1") -> bool:
        """Delete features for a sample."""
        if not self.enabled or not self.client:
            return False
        
        try:
            key = self._make_key(sample_id, version)
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete features: {e}")
            return False
    
    def exists(self, sample_id: str, version: str = "v1") -> bool:
        """Check if features exist for a sample."""
        if not self.enabled or not self.client:
            return False
        
        try:
            key = self._make_key(sample_id, version)
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check existence: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        if not self.enabled or not self.client:
            return {"enabled": False}
        
        try:
            info = self.client.info()
            return {
                "enabled": True,
                "total_keys": info.get("db0", {}).get("keys", 0),
                "memory_used": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def clear_all(self, version: Optional[str] = None):
        """
        Clear all features (use with caution!).
        
        Args:
            version: If specified, only clear features of this version
        """
        if not self.enabled or not self.client:
            return
        
        try:
            if version:
                pattern = f"features:{version}:*"
                keys = self.client.keys(pattern)
                if keys:
                    self.client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} keys for version {version}")
            else:
                self.client.flushdb()
                logger.warning("Cleared all features from store")
        except Exception as e:
            logger.error(f"Failed to clear features: {e}")
    
    def _make_key(self, sample_id: str, version: str) -> str:
        """Generate Redis key for sample."""
        return f"features:{version}:{sample_id}"
    
    def _update_metadata(self, sample_id: str, version: str, feature_count: int):
        """Update metadata for tracking."""
        try:
            meta_key = f"meta:{version}:stats"
            self.client.hincrby(meta_key, "total_samples", 1)
            self.client.hincrby(meta_key, "total_features", feature_count)
            self.client.hset(meta_key, "last_updated", datetime.utcnow().isoformat())
        except Exception as e:
            logger.debug(f"Failed to update metadata: {e}")
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()


# Convenience functions
def cache_features(
    sample_id: str,
    features: Dict[str, Any],
    store: Optional[FeatureStore] = None
) -> bool:
    """
    Cache features with default store.
    
    Args:
        sample_id: Sample identifier
        features: Features dictionary
        store: FeatureStore instance (creates new if None)
        
    Returns:
        True if successful
    """
    if store is None:
        store = FeatureStore()
    
    return store.store_features(sample_id, features)


def get_cached_features(
    sample_id: str,
    store: Optional[FeatureStore] = None
) -> Optional[Dict[str, Any]]:
    """
    Get cached features with default store.
    
    Args:
        sample_id: Sample identifier
        store: FeatureStore instance (creates new if None)
        
    Returns:
        Features dictionary or None
    """
    if store is None:
        store = FeatureStore()
    
    return store.get_features(sample_id)


if __name__ == "__main__":
    # Example usage
    store = FeatureStore(host="localhost", port=6379)
    
    # Store features
    sample_features = {
        "feature_1": 0.5,
        "feature_2": 1.2,
        "feature_3": -0.3
    }
    
    success = store.store_features("sample_001", sample_features)
    print(f"Stored: {success}")
    
    # Retrieve features
    retrieved = store.get_features("sample_001")
    print(f"Retrieved: {retrieved}")
    
    # Get stats
    stats = store.get_stats()
    print(f"Stats: {stats}")
