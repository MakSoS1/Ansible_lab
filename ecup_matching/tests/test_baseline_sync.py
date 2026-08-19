from ecup_matching.baseline_sync import BASELINE_SOURCES


def test_official_baseline_sources_are_mirrored_under_baselines_directory():
    assert [source.name for source in BASELINE_SOURCES] == [
        "baselines/matching-baseline-submit.zip",
        "baselines/matching-baseline-lightweight.zip",
    ]
    assert [source.url for source in BASELINE_SOURCES] == [
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matching-baseline-submit.zip",
        "https://storage.yandexcloud.net/ozon-ecup-2026/Matching/matching-baseline-lightweight.zip",
    ]
