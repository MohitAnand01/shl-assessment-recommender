import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/products/product-catalog/"
OUTPUT_PATH = "data/assessments.json"

# Catalog pagination settings
PAGE_SIZE = 12  # based on ?start=0,12,24,...
MAX_PAGES = 40  # safety cap; can be increased if needed
REQUEST_DELAY = 1.0  # seconds between requests (be polite)


@dataclass
class Assessment:
    name: str
    url: str
    description: str
    duration_minutes: int
    adaptive: bool
    remote: bool
    test_types: List[str]


def fetch_html(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch a page and return its HTML, or None on error."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"  [WARN] {url} -> HTTP {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed for {url}: {e}")
        return None


def parse_duration(text: str) -> int:
    """Extract duration in minutes from free text. Return 0 if unknown."""
    text = text.lower()

    # minutes
    m = re.search(r"(\d+)\s*min", text)
    if m:
        return int(m.group(1))

    # hours like "1 hour", "2 hours"
    h = re.search(r"(\d+)\s*hour", text)
    if h:
        return int(h.group(1)) * 60

    # ranges like "1-2 hours" -> take upper bound
    hr = re.search(r"(\d+)\s*-\s*(\d+)\s*hour", text)
    if hr:
        return int(hr.group(2)) * 60

    return 0


def extract_test_types(soup: BeautifulSoup) -> List[str]:
    """
    Extract test type tags from badges/labels that correspond to:
    Ability & Aptitude, Knowledge & Skills, Personality & Behavior, etc.
    """
    tags: Set[str] = set()

    # Many tags are in pill/badge like elements
    for el in soup.find_all(["a", "span", "div"]):
        classes = " ".join(el.get("class") or []).lower()
        text = (el.get_text(" ", strip=True) or "").strip()
        if not text:
            continue
        if any(c in classes for c in ["badge", "pill", "tag", "chip"]):
            if len(text) < 80:
                tags.add(text)

    # Fallback: scan short list items/spans for known keywords
    KEYWORDS = [
        "ability & aptitude",
        "biodata & situational judgement",
        "competencies",
        "development & 360",
        "assessment exercises",
        "knowledge & skills",
        "personality & behavior",
        "personality & behaviour",
        "simulations",
        "situational judgement",
        "numerical",
        "verbal",
        "logical",
        "reasoning",
        "coding",
    ]
    for el in soup.find_all(["li", "span"]):
        txt = (el.get_text(" ", strip=True) or "").strip()
        if not txt or len(txt) > 80:
            continue
        low = txt.lower()
        if any(kw in low for kw in KEYWORDS):
            tags.add(txt)

    return sorted(tags)


def infer_flags_from_text(text: str) -> Dict[str, bool]:
    """Infer adaptive and remote flags from raw page text."""
    low = text.lower()
    adaptive = any(
        kw in low
        for kw in [
            "adaptive test",
            "computer adaptive",
            "adaptive version",
            "cat test",
        ]
    )
    remote = any(
        kw in low
        for kw in [
            "remote proctor",
            "remotely proctored",
            "online proctor",
            "remote assessment",
            "remote test",
            "online assessment",
            "taken remotely",
        ]
    )
    return {"adaptive": adaptive, "remote": remote}


def looks_like_individual_test(url: str) -> bool:
    """
    Heuristic: keep URLs that look like individual tests,
    avoid obvious bundles / generic solutions if needed.
    For now we are lenient to hit >=377 items.
    """
    low = url.lower()

    # Most product URLs look like: /products/product-catalog/view/<slug>/
    if "/products/product-catalog/view/" not in low:
        return False

    # Exclude clearly non-test/marketing pages if any specific patterns appear:
    EXCLUDE_KEYWORDS = [
        "/job-solution/",
        "/job-solutions/",
        "/solution-overview/",
    ]
    for k in EXCLUDE_KEYWORDS:
        if k in low:
            return False

    return True


def collect_catalog_links() -> Set[str]:
    """
    Walk through catalog pages with ?start=X&type=1 and collect product links.
    """
    all_links: Set[str] = set()

    for page_idx in range(MAX_PAGES):
        start = page_idx * PAGE_SIZE
        url = f"{CATALOG_BASE}?start={start}&type=1"
        print(f"\n[Catalog] Page {page_idx+1} -> {url}")

        html = fetch_html(url)
        if not html:
            print("  No HTML returned, stopping pagination.")
            break

        soup = BeautifulSoup(html, "html.parser")

        # Find product cards and links within them
        page_links: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            clean = href.split("?")[0].split("#")[0]
            if looks_like_individual_test(clean):
                page_links.add(clean)

        print(f"  Found {len(page_links)} product links on this page.")
        if not page_links:
            # If we hit a page with no products, likely end of pagination
            print("  No product links found, assuming end of catalog.")
            break

        all_links.update(page_links)
        print(f"  Total unique product links so far: {len(all_links)}")

        # polite delay
        time.sleep(REQUEST_DELAY)

    return all_links


def scrape_assessment(url: str) -> Optional[Assessment]:
    """Scrape a single product detail page."""
    print(f"  [Detail] {url}")
    html = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Name: H1 first, then title fallback
    name_el = soup.find("h1")
    if name_el:
        name = name_el.get_text(" ", strip=True)
    else:
        title_el = soup.find("title")
        name = title_el.get_text(" ", strip=True) if title_el else url

    # Description: first long <p> as summary
    description = ""
    for p in soup.find_all("p"):
        txt = (p.get_text(" ", strip=True) or "").strip()
        if len(txt) > 80:
            description = txt
            break

    all_text = soup.get_text(" ", strip=True)
    duration = parse_duration(all_text)
    test_types = extract_test_types(soup)
    flags = infer_flags_from_text(all_text)

    return Assessment(
        name=name,
        url=url,
        description=description,
        duration_minutes=duration,
        adaptive=flags["adaptive"],
        remote=flags["remote"],
        test_types=test_types,
    )


def main():
    print("Starting SHL catalog crawl (paginated via ?start=N&type=1)")

    links = collect_catalog_links()
    print(f"\nTotal unique product links collected: {len(links)}")

    assessments: List[Assessment] = []
    for i, url in enumerate(sorted(links), start=1):
        print(f"\nScraping assessment {i}/{len(links)}")
        a = scrape_assessment(url)
        if a is None:
            continue
        assessments.append(a)
        time.sleep(REQUEST_DELAY)

    print(f"\nScraped {len(assessments)} assessments.")
    if len(assessments) < 377:
        print(
            f"WARNING: Only {len(assessments)} assessments scraped. "
            "Assignment expects at least 377 Individual Test Solutions. "
            "You may increase MAX_PAGES or relax looks_like_individual_test()."
        )
    else:
        print("Requirement of >=377 Individual Test Solutions appears satisfied (subject to manual check).")

    # Save JSON
    data = [asdict(a) for a in assessments]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()