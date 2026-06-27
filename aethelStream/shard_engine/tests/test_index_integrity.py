"""Test 5 — shard_index.json integrity checks."""
from __future__ import annotations
import pytest

def test_index_all_files_exist(tiny_shard_dir, tiny_shard_index):
    """verify_index_integrity must pass when all files exist and offsets are valid."""
    from python.pipeline.verifier import verify_index_integrity
    verify_index_integrity(tiny_shard_index, tiny_shard_dir)

def test_index_missing_file_raises(tmp_path, tiny_shard_index):
    """verify_index_integrity must raise AssertionError if a shard file is missing."""
    with pytest.raises(AssertionError, match="missing"):
        from python.pipeline.verifier import verify_index_integrity
        verify_index_integrity(tiny_shard_index, tmp_path)

def test_index_offset_out_of_bounds_raises(tiny_shard_dir, tiny_shard_index):
    """verify_index_integrity must raise AssertionError if offset exceeds file size."""
    import copy
    bad_index = copy.deepcopy(tiny_shard_index)
    param_name = list(bad_index.keys())[0]
    bad_index[param_name]["byte_offset"] = 999999
    bad_index[param_name]["byte_length"] = 999999
    from python.pipeline.verifier import verify_index_integrity
    with pytest.raises(AssertionError, match="offset"):
        verify_index_integrity(bad_index, tiny_shard_dir)
