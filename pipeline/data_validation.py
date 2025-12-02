"""Data validation, schema enforcement, lineage and data versioning utilities.

This module provides a light-weight schema validator that uses `pandera` if available,
otherwise falls back to basic pandas-based checks. It also computes a deterministic
content hash for datasets to enable data-version tracking and lineage logging.

Key classes:
- SchemaValidator: validate DataFrame structure and basic constraints.
- DataLineage: compute dataset hashes and metadata (source, timestamp, schema_hash).
- DataVersionStore: simple local store for mapping data hash -> metadata; integrates with MLflow when available.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import time
import json
import hashlib

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except Exception:
    pd = None

# Try to import pandera for richer schema validation
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema
    _HAS_PANDERA = True
except Exception:
    _HAS_PANDERA = False


class SchemaValidator:
    """Validate pandas DataFrame according to a declarative schema.

    Example schema definition:
        schema = {
            'feature1': {'dtype': 'float', 'nullable': False, 'min': -10, 'max': 10},
            'feature2': {'dtype': 'int', 'nullable': True},
            'label': {'dtype': 'int', 'nullable': False}
        }
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        self.schema = schema
        self._prepare()

    def _prepare(self):
        if _HAS_PANDERA:
            cols = {}
            for name, conf in self.schema.items():
                dtype = conf.get('dtype', 'float')
                nullable = conf.get('nullable', True)
                checks = []
                if 'min' in conf:
                    checks.append(pa.Check(lambda s: s >= conf['min']))
                if 'max' in conf:
                    checks.append(pa.Check(lambda s: s <= conf['max']))
                cols[name] = Column(dtype, nullable=nullable, checks=checks)
            self._pa_schema = DataFrameSchema(cols)
        else:
            self._pa_schema = None

    def validate(self, df) -> bool:
        """Validate DataFrame; returns True if valid, raises on failure."""
        if pd is None:
            raise RuntimeError('pandas is required for data validation')

        if not isinstance(df, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')

        # Quick column presence check
        missing = [c for c in self.schema.keys() if c not in df.columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

        if _HAS_PANDERA and self._pa_schema is not None:
            self._pa_schema.validate(df, lazy=True)
            return True

        # Fallback checks: dtype and min/max
        for name, conf in self.schema.items():
            series = df[name]
            if not conf.get('nullable', True):
                if series.isnull().any():
                    raise ValueError(f'Column {name} contains nulls but is not nullable')
            dtype = conf.get('dtype')
            if dtype:
                # Minimal dtype checks
                if dtype in ('int', 'integer') and not pd.api.types.is_integer_dtype(series):
                    raise ValueError(f'Column {name} expected integer dtype')
                if dtype in ('float', 'double') and not pd.api.types.is_float_dtype(series):
                    # allow integer as float-compatible
                    if not pd.api.types.is_integer_dtype(series):
                        raise ValueError(f'Column {name} expected float dtype')
            if 'min' in conf:
                if (series < conf['min']).any():
                    raise ValueError(f'Column {name} has values below min {conf["min"]}')
            if 'max' in conf:
                if (series > conf['max']).any():
                    raise ValueError(f'Column {name} has values above max {conf["max"]}')
        return True


class DataLineage:
    """Compute deterministic fingerprint of a dataset and store basic lineage metadata."""

    @staticmethod
    def dataframe_hash(df, normalize: bool = True) -> str:
        """Compute SHA256 hash for a DataFrame's contents. This is deterministic for the same
        data and column order. If `normalize` is True, we'll sort columns and reset index.
        """
        if pd is None:
            raise RuntimeError('pandas is required for data hashing')
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')

        if normalize:
            d = df.copy()
            d = d.sort_index(axis=1)
            d = d.reset_index(drop=True)
        else:
            d = df

        # Convert to CSV bytes deterministically
        csv_bytes = d.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(csv_bytes).hexdigest()

    @staticmethod
    def schema_hash(schema: Dict[str, Any]) -> str:
        serial = json.dumps(schema, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serial).hexdigest()

    @staticmethod
    def build_lineage_record(
        source: str,
        df,
        schema: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            'source': source,
            'timestamp': int(time.time()),
            'rows': int(len(df)) if pd is not None and isinstance(df, pd.DataFrame) else None,
        }
        record['data_hash'] = DataLineage.dataframe_hash(df) if pd is not None else None
        if schema is not None:
            record['schema_hash'] = DataLineage.schema_hash(schema)
        if extra:
            record['extra'] = extra
        return record


class DataVersionStore:
    """Local data version store that maps data_hash -> metadata. Optionally integrate with MLflow.

    This is intentionally minimal: it stores JSON files under `.data_versions/` and can also
    call MLflow to log a dataset artifact or tag a run with the data hash.
    """

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = Path(store_dir or Path('data_versions'))
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        try:
            import mlflow

            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    def record(self, data_hash: str, metadata: Dict[str, Any]):
        path = self.store_dir / f'{data_hash}.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f'Data version recorded: {path}')
        # If mlflow available, log as tag on active run
        if self._mlflow and self._mlflow.active_run() is not None:
            try:
                self._mlflow.set_tag('data_hash', data_hash)
                self._mlflow.log_text(json.dumps(metadata), artifact_file=f'data_versions/{data_hash}.json')
            except Exception as e:
                logger.debug(f'Failed to log data version to mlflow: {e}')

    def get(self, data_hash: str) -> Optional[Dict[str, Any]]:
        path = self.store_dir / f'{data_hash}.json'
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


# Small helper to integrate validation + lineage
class DataValidatorAndVersioner:
    def __init__(self, schema: Dict[str, Any], source: str, store_dir: Optional[Path] = None):
        self.validator = SchemaValidator(schema)
        self.source = source
        self.store = DataVersionStore(store_dir)
        self.schema = schema

    def validate_and_record(self, df) -> Dict[str, Any]:
        # Validate
        self.validator.validate(df)
        # Build lineage record
        rec = DataLineage.build_lineage_record(self.source, df, self.schema)
        # Record
        self.store.record(rec['data_hash'], rec)
        return rec
