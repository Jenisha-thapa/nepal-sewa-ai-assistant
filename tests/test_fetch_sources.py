"""Unit tests for pure helper functions in src.ingestion.fetch_sources.

Deliberately does not test fetch_one_source() / run() with real network
calls — those need mocking (e.g. with `responses` or `unittest.mock`) before
CI can run them offline. Add that once the ingestion module stabilizes.
"""

from src.ingestion.fetch_sources import guess_extension, load_domain_config, slugify
from pathlib import Path


def test_slugify_basic():
    assert slugify("Department of Land Management and Archive") == \
        "Department-of-Land-Management-and-Archive"


def test_slugify_strips_url_scheme():
    assert slugify("https://donidcr.gov.np/pages/faq") == "donidcr.gov.np-pages-faq"


def test_slugify_truncates_long_input():
    long_text = "a" * 200
    assert len(slugify(long_text, max_length=50)) == 50


def test_guess_extension_known_types():
    assert guess_extension("text/html; charset=utf-8") == ".html"
    assert guess_extension("application/pdf") == ".pdf"
    assert guess_extension(None) == ".bin"
    assert guess_extension("application/octet-stream") == ".bin"


def test_load_domain_config_has_four_domains():
    config = load_domain_config(Path("config/domains.yaml"))
    domain_ids = {d["id"] for d in config["domains"]}
    assert domain_ids == {
        "identity_civil_registration",
        "land_property",
        "employment_labour",
        "health_social_welfare",
    }


def test_every_source_has_required_fields():
    config = load_domain_config(Path("config/domains.yaml"))
    for domain in config["domains"]:
        for source in domain["sources"]:
            assert "name" in source
            assert "url" in source
            assert source["url"].startswith("http")
