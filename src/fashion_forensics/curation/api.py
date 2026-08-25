"""FastAPI backend for the trend curation review app.

Run with:
    .venv/bin/uvicorn fashion_forensics.curation.api:app --reload --port 8010

The Streamlit review app (fashion_forensics.curation.review_app) talks to
this service over HTTP so scraping/storage can run headless (e.g. on a
schedule) independent of whether anyone has the review UI open.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fashion_forensics.curation import store
from fashion_forensics.curation.scraper import TrendSiteScraper

app = FastAPI(title="Trend Curation API")


class ApproveRequest(BaseModel):
    trend: str
    reviewer: str | None = None


class RejectRequest(BaseModel):
    reviewer: str | None = None
    reason: str | None = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    trend_hint: str | None = None


class ScrapeUrlRequest(BaseModel):
    url: str
    name: str | None = None
    trend_hint: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/trends")
def trends() -> dict[str, list[str]]:
    return {"trends": store.list_known_trends()}


@app.get("/stats")
def stats() -> dict:
    return store.stats()


@app.get("/candidates")
def candidates(status: str = "pending", trend_hint: str | None = None) -> dict[str, list[dict]]:
    return {"candidates": store.list_candidates(status=status, trend_hint=trend_hint)}


@app.get("/candidates/{candidate_id}/image")
def candidate_image(candidate_id: str) -> FileResponse:
    path = store.image_path(candidate_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.post("/candidates/{candidate_id}/approve")
def approve_candidate(candidate_id: str, body: ApproveRequest) -> dict:
    try:
        return store.approve(candidate_id, body.trend, reviewer=body.reviewer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Candidate not found") from e
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/candidates/{candidate_id}/reject")
def reject_candidate(candidate_id: str, body: RejectRequest) -> dict:
    try:
        return store.reject(candidate_id, reviewer=body.reviewer, reason=body.reason)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Candidate not found") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/scrape")
def trigger_scrape() -> dict:
    """Run the configured trend-site scrape synchronously and return per-source counts."""
    scraper = TrendSiteScraper()
    return scraper.run()


@app.post("/scrape-url")
def scrape_single_url(body: ScrapeUrlRequest) -> dict:
    """Scrape one page on demand (e.g. a pasted Pinterest board/pin link)."""
    scraper = TrendSiteScraper()
    return scraper.scrape_url(body.url, name=body.name, trend_hint=body.trend_hint)


@app.post("/search")
def search_images(body: SearchRequest) -> dict:
    """Run a free-text image search (Google CSE) and queue results for review."""
    from fashion_forensics.curation.image_search import GoogleImageSearchProvider

    try:
        provider = GoogleImageSearchProvider()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return provider.run(body.query, max_results=body.max_results, trend_hint=body.trend_hint)
