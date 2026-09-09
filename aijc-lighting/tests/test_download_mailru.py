import pytest

from scripts.download_mailru import (
    build_download_candidates,
    build_download_url,
    extract_filename_from_html,
    extract_public_id,
)


def test_extract_public_id():
    assert extract_public_id('https://cloud.mail.ru/public/GCsv/1BXmZPEBj') == 'GCsv/1BXmZPEBj'


def test_extract_public_id_rejects_other_hosts():
    with pytest.raises(ValueError):
        extract_public_id('https://example.com/public/GCsv/1BXmZPEBj')


def test_extract_filename_from_html_title():
    html = '<html><head><title>data_освещённость.zip / Облако Mail</title></head></html>'
    assert extract_filename_from_html(html) == 'data_освещённость.zip'


def test_build_download_url_uses_weblink_get_endpoint():
    dispatcher = {'body': {'weblink_get': [{'url': 'https://storage.example/weblink/get'}]}}
    url = build_download_url(
        'https://cloud.mail.ru/public/GCsv/1BXmZPEBj',
        dispatcher,
        {'body': {'token': 'abc123'}},
    )
    assert url.startswith('https://storage.example/weblink/get/GCsv/1BXmZPEBj')
    assert 'key=abc123' in url


def test_candidates_include_plain_and_filename_variants():
    dispatcher = {'body': {'weblink_get': [{'url': 'https://storage.example/weblink/get'}]}}
    urls = build_download_candidates(
        'https://cloud.mail.ru/public/GCsv/1BXmZPEBj',
        dispatcher,
        {'body': {'token': 'abc123'}},
        filename='data_освещённость.zip',
    )
    assert any('/GCsv/1BXmZPEBj?key=abc123' in value for value in urls)
    assert any(
        '/GCsv/1BXmZPEBj/data_%D0%BE%D1%81%D0%B2%D0%B5%D1%89%D1%91%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D1%8C.zip?key=abc123'
        in value
        for value in urls
    )
