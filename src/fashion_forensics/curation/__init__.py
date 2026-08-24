"""Trend curation: scrape trend-site images, review, and promote approvals
into the labeled reference corpus (data/02_reference_corpus/labeled/).

    fashion_forensics.curation.scraper  — pulls candidate images into the review queue
    fashion_forensics.curation.store    — queue persistence + approve/reject
    fashion_forensics.curation.api      — FastAPI backend for the review app
"""
