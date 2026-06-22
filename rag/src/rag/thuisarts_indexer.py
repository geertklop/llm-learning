"""Index Thuisarts.nl triage guidelines into pgvector."""

import logging
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx
import ollama
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from rag.database.crud import insert_guideline

logger = logging.getLogger(__name__)

_SITEMAP_PAGES = [
    "https://www.thuisarts.nl/sitemap.xml?page=1",
    "https://www.thuisarts.nl/sitemap.xml?page=2",
]

_USER_AGENT = "llm-learning/1.0 (educational project)"

# Seconds between HTTP requests — keeps load on thuisarts.nl negligible.
_REQUEST_DELAY = 0.2

# First path segments that are never medical articles.
_NON_ARTICLE_SLUGS = {
    "over-thuisarts",
    "nieuws",
    "overzicht",
    "vertalingen-op-thuisarts",
    "samenwerken",
    "privacy-en-cookies-op-thuisarts",
    "disclaimer",
    "spoed-wie-bel-je",
    "contact",
    "zoeken",
    "anatomisch",
    "dutch-healthcare",
    "toegankelijkheid",
    "thuisarts-is-voor-iedereen",
    "gebruiksvoorwaarden",
    "deel-via-email",
}


def ingest_guidelines(session: Session, embed_model: str) -> None:
    """
    Scrape Thuisarts.nl triage pages and index them into pgvector.

    Discovers all patient-situation sub-pages from the sitemap, extracts the
    "Wanneer bel je de huisarts" section from each, embeds title + context
    together, and upserts into the guidelines table.

    Parameters
    ----------
    session
        An open SQLAlchemy session. Committed in batches and finally at the end.
    embed_model
        Ollama model name used to produce embedding vectors.
    """
    with httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=30,
    ) as client:
        urls = _discover_article_urls(client)
        logger.info("Discovered %d candidate triage article URLs", len(urls))

        indexed = 0
        for index, url in enumerate(urls):
            try:
                html = _fetch_page(client, url)
                chunks = _parse_article_chunks(html, url)
            except httpx.HTTPStatusError as exc:
                logger.warning("HTTP %s for %s — skipping", exc.response.status_code, url)
                time.sleep(_REQUEST_DELAY)
                continue
            except Exception as exc:
                logger.warning("Failed to fetch/parse %s: %s — skipping", url, exc)
                time.sleep(_REQUEST_DELAY)
                continue

            if not chunks:
                logger.debug("No triage section found in %s — skipping", url)
                time.sleep(_REQUEST_DELAY)
                continue

            for chunk in chunks:
                # Each chunk is one urgency sub-section. Include the heading
                # so the vector captures the urgency signal alongside criteria.
                text_to_embed = f"{chunk['title']}\n\n{chunk['context']}"
                embedding = _embed(text_to_embed, embed_model)
                insert_guideline(session=session, embedding=embedding, **chunk)
                indexed += 1

            if indexed % 20 == 0:
                session.commit()

            logger.info("[%d/%d] indexed %d chunks from %s", index + 1, len(urls), len(chunks), url)
            time.sleep(_REQUEST_DELAY)

    session.commit()
    logger.info("Guideline indexing complete. %d articles indexed.", indexed)


def _discover_article_urls(client: httpx.Client) -> list[str]:
    """
    Parse both sitemap pages and return URLs of patient-situation sub-pages.

    Patient-situation pages have exactly 2 path segments (e.g.
    /buikpijn/ik-heb-buikpijn) and a first segment that is not in the
    known non-article slug list.

    Parameters
    ----------
    client
        An open httpx client to reuse for sitemap requests.

    Returns
    -------
    Deduplicated list of candidate article URLs.
    """
    urls: list[str] = []
    for sitemap_url in _SITEMAP_PAGES:
        response = client.get(sitemap_url)
        response.raise_for_status()
        # The sitemap uses the standard XML Sitemap namespace.
        root = ET.fromstring(response.text)
        for element in root.iter():
            if element.tag.endswith("}loc") or element.tag == "loc":
                loc = (element.text or "").strip()
                if _is_article_url(loc):
                    urls.append(loc)
        time.sleep(_REQUEST_DELAY)
    return urls


def _is_article_url(url: str) -> bool:
    """
    Return True if the URL looks like a patient-situation sub-page.

    Parameters
    ----------
    url
        Absolute URL from the sitemap.
    """
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    if len(segments) != 2:
        return False
    first_segment = segments[0]
    return first_segment not in _NON_ARTICLE_SLUGS


def _fetch_page(client: httpx.Client, url: str) -> str:
    """
    Fetch a page and return its HTML content.

    Parameters
    ----------
    client
        An open httpx client.
    url
        The page URL to fetch.

    Returns
    -------
    Raw HTML string.
    """
    response = client.get(url)
    response.raise_for_status()
    return response.text


def _parse_article_chunks(html: str, url: str) -> list[dict]:
    """
    Split a Thuisarts triage page into one chunk per urgency sub-section.

    The "Wanneer bel je de huisarts" h2 is followed by a nested structure:
    ``div > article > div > div.text-formatted``. Inside that container the
    content is split by ``<h4>`` headings ("Vandaag bellen", "Wel bellen").
    The initial block before the first h4 is the "Spoed: bel direct" section,
    which has no heading tag of its own.

    Parameters
    ----------
    html
        Raw HTML of the article page.
    url
        The canonical page URL, used to derive the slug and chunk URLs.

    Returns
    -------
    List of dicts, each with keys ``url``, ``title``, ``slug``,
    ``urgency_hint``, and ``context``. Empty list if no triage section found.
    """
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1")
    if title_tag is None:
        return []
    title = title_tag.get_text(strip=True)

    triage_heading = None
    for heading in soup.find_all("h2"):
        if "wanneer bel" in heading.get_text().lower():
            triage_heading = heading
            break

    if triage_heading is None:
        return []

    # Content lives inside: sibling div > article > div > div.text-formatted
    content_div = None
    for sibling in triage_heading.next_siblings:
        if getattr(sibling, "name", None) is not None:
            content_div = sibling.find(
                "div", class_=lambda c: c is not None and "text-formatted" in c
            )
            break

    if content_div is None:
        return []

    slug = urlparse(url).path.split("/")[1]
    chunks: list[dict] = []
    current_subheading: str | None = None
    current_parts: list[str] = []

    for child in content_div.children:
        tag_name = getattr(child, "name", None)
        if tag_name is None:
            continue

        if tag_name == "h4":
            if current_parts:
                heading = current_subheading or _infer_spoed_heading(current_parts)
                chunks.append(_make_chunk(url, title, slug, heading, current_parts))
            current_subheading = child.get_text(strip=True)
            current_parts = []
        else:
            text = child.get_text(separator=" ", strip=True)
            if text:
                current_parts.append(text)

    if current_parts:
        heading = current_subheading or _infer_spoed_heading(current_parts)
        chunks.append(_make_chunk(url, title, slug, heading, current_parts))

    return chunks


def _infer_spoed_heading(parts: list[str]) -> str:
    """
    Infer the heading for the implicit first chunk (before the first h4).

    The "Spoed: bel direct" section has no h4 — its urgency signal appears
    inside the first paragraph text. Fall back to a generic label if not found.

    Parameters
    ----------
    parts
        Text blocks collected so far.

    Returns
    -------
    "Spoed: bel direct" if the text contains "spoed", else "Algemeen".
    """
    combined = " ".join(parts).lower()
    return "Spoed: bel direct" if "spoed" in combined else "Algemeen"


def _make_chunk(
    url: str,
    title: str,
    slug: str,
    subheading: str,
    parts: list[str],
) -> dict:
    """
    Build a single chunk dict from a parsed urgency sub-section.

    Parameters
    ----------
    url
        Base article URL.
    title
        Article h1 title.
    slug
        Condition slug from the URL path.
    subheading
        Text of the h3/h4 urgency heading.
    parts
        Text blocks collected under that heading.

    Returns
    -------
    Dict ready to pass to ``insert_guideline``.
    """
    # Prefix the context with the sub-heading so the embedding captures the
    # urgency signal ("Spoed: bel direct") alongside the criteria text.
    context = f"{subheading}\n\n" + "\n".join(parts)
    urgency_hint = _classify_urgency(subheading)
    # Append a URL fragment derived from the heading for row uniqueness.
    fragment = subheading.lower().replace(" ", "-").replace(":", "")[:30]
    chunk_url = f"{url}#{fragment}"
    return {
        "url": chunk_url,
        "title": title,
        "slug": slug,
        "urgency_hint": urgency_hint,
        "context": context,
    }


def _classify_urgency(heading: str) -> str | None:
    """
    Map a Thuisarts urgency sub-heading to a triage level.

    Parameters
    ----------
    heading
        The h3/h4 heading text from the triage section.

    Returns
    -------
    ``"red"``, ``"orange"``, ``"yellow"``, or ``None`` if unrecognised.
    """
    lower = heading.lower()
    if "geen spoed" in lower or "wel bellen" in lower:
        return "yellow"
    if "spoed" in lower:
        return "red"
    if "vandaag" in lower:
        return "orange"
    return None


def _embed(text: str, model: str) -> list[float]:
    """
    Produce an embedding vector for a piece of text.

    Parameters
    ----------
    text
        The text to embed.
    model
        Ollama model name used for embedding.

    Returns
    -------
    A list of floats representing the embedding vector.
    """
    response = ollama.embed(model=model, input=text)
    return response.embeddings[0]
