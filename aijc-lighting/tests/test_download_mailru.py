import pytest

from scripts.download_mailru import extract_public_id, build_download_url


def test_extract_public_id():
    assert extract_public_id('https://cloud.mail.ru/public/GCsv/1BXmZPEBj') == 'GCsv/1BXmZPEBj'


def test_extract_public_id_rejects_other_hosts():
    with pytest.raises(ValueError):
        extract_public_id('https://example.com/public/GCsv/1BXmZPEBj')


def test_build_download_url_uses_weblink_get_endpoint():
    dispatcher = {'body': {'weblink_get': [{'url': 'https://storage.example/weblink/get'}]}}
    url = build_download_url(
        'https://cloud.mail.ru/public/GCsv/1BXmZPEBj',
        dispatcher,
        {'body': {'token': 'abc123'}},
    )
    assert url.startswith('https://storage.example/weblink/get/GCsv/1BXmZPEBj')
    assert 'key=abc123' in url
