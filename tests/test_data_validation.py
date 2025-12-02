"""Tests for data validation, lineage and data versioning."""
import pandas as pd
from pathlib import Path
import shutil

from pipeline.data_validation import (
    SchemaValidator,
    DataLineage,
    DataVersionStore,
    DataValidatorAndVersioner,
)


def test_schema_validator_pass():
    schema = {
        'a': {'dtype': 'float', 'nullable': False, 'min': -1.0, 'max': 1.0},
        'b': {'dtype': 'int', 'nullable': False},
    }
    df = pd.DataFrame({'a': [0.0, 0.5], 'b': [1, 2]})
    v = SchemaValidator(schema)
    assert v.validate(df) is True


def test_schema_validator_missing_column():
    schema = {'a': {'dtype': 'float'}}
    df = pd.DataFrame({'b': [1]})
    v = SchemaValidator(schema)
    try:
        v.validate(df)
        assert False, "Expected ValueError for missing column"
    except ValueError:
        assert True


def test_data_hash_and_store(tmp_path):
    df = pd.DataFrame({'x': [1, 2, 3]})
    h = DataLineage.dataframe_hash(df)
    store = DataVersionStore(tmp_path / 'store')

    meta = {'source': 'unit_test', 'rows': 3}
    store.record(h, meta)
    loaded = store.get(h)
    assert loaded is not None
    assert loaded['source'] == 'unit_test'


def test_validator_and_versioner(tmp_path):
    schema = {'x': {'dtype': 'int', 'nullable': False}}
    df = pd.DataFrame({'x': [1, 2, 3]})
    dv = DataValidatorAndVersioner(schema, source='test', store_dir=tmp_path / 'dvs')
    rec = dv.validate_and_record(df)
    assert 'data_hash' in rec
    # file exists
    assert (tmp_path / 'dvs' / f"{rec['data_hash']}.json").exists()
