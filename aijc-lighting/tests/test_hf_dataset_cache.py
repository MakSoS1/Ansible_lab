import pytest

from scripts.hf_dataset_cache import dataset_repo_id


def test_dataset_repo_id_uses_authenticated_owner():
    assert dataset_repo_id('maksos') == 'maksos/aijc-lighting-official-data'


def test_dataset_repo_id_rejects_empty_owner():
    with pytest.raises(ValueError):
        dataset_repo_id('   ')
