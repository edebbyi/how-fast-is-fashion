"""Run the trend-site curation scraper from the command line.

Pulls candidate images from every source in configs/curation.yaml into the
review queue (data/04_curation/). Nothing touches the labeled reference
corpus here — review and approve via the Streamlit app:

    .venv/bin/uvicorn fashion_forensics.curation.api:app --port 8010
    .venv/bin/streamlit run src/fashion_forensics/curation/review_app.py

Usage:
    .venv/bin/python scripts/scrape_trend_sites.py
"""

from __future__ import annotations

from fashion_forensics.curation.scraper import TrendSiteScraper


def main() -> None:
    scraper = TrendSiteScraper()
    summary = scraper.run()
    for source_name, counts in summary.items():
        print(f"{source_name}: {counts}")


if __name__ == "__main__":
    main()
