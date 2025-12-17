import json
import re
import time
from dataclasses import asdict, dataclass
from typing import List, Dict, Set

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


# TODO: PUT YOUR ACTUAL CATALOG URL(S) HERE.
# Open SHL in your browser, go to the assessment catalog listing page,
# copy the URL, and paste it below.
CATALOG_URLS = [
    # Example placeholders – REPLACE these with the real URLs you see:
    # "https://www.shl.com/en/assessments/product-catalog/",
    # "https://www.shl.com/en/solutions/products/product-catalog/"
]

OUTPUT_PATH = "data/assessments.json"
MAX_PRODUCTS_HINT = 400  # just for logging


@dataclass
class Assessment:
    name: str
    url: str
    description: str
    duration_minutes: int
    adaptive: bool
    remote: bool
    test_types: List[str]


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Chrome WebDriver with sensible defaults."""
    options = webdriver.ChromeOptions()
    if headless:
        # new headless mode for recent Chrome
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


def polite_sleep(seconds: float):
    """Throttle requests a bit to be polite to the server."""
    time.sleep(seconds)


def scroll_to_bottom(driver: webdriver.Chrome, max_rounds: int = 30, wait: float = 2.0):
    """
    Scrolls to the bottom of the page, trying to trigger lazy-loading/infinite scroll.
    Stops when the height stops increasing or max_rounds is reached.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    rounds = 0

    while rounds < max_rounds:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        polite_sleep(wait)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        rounds += 1


def click_load_more_if_present(driver: webdriver.Chrome) -> bool:
    """
    Tries to click a 'Load more' or 'Show more' button, if present.
    Returns True if clicked, False if not found.
    """
    xpaths = [
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'load more')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'show more')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'load more')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'show more')]",
    ]
    for xp in xpaths:
        try:
            btn = driver.find_element(By.XPATH, xp)
            driver.execute_script("arguments[0].click();", btn)
            polite_sleep(3.0)
            return True
        except NoSuchElementException:
            continue
    return False


def collect_product_links_from_catalog(driver: webdriver.Chrome, url: str) -> Set[str]:
    """
    Opens a catalog listing page and returns a set of product detail URLs.
    Tries scrolling, clicking 'Load more', etc.
    """
    links: Set[str] = set()
    print(f"\n[Catalog] Loading: {url}")
    driver.get(url)
    polite_sleep(5.0)

    # Try a few rounds of scrolling + 'Load more' clicks.
    for i in range(10):
        print(f"  Scroll/Load round {i+1}/10")
        scroll_to_bottom(driver, max_rounds=3, wait=2.0)
        if not click_load_more_if_present(driver):
            # no load-more button found; break if height stops changing
            # scroll_to_bottom already did that
            break

    # Now parse the DOM and collect product links
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http"):
            continue
        # Heuristic: SHL product URLs contain "/product-catalog/view/"
        if "/product-catalog/view/" in href:
            # normalize by dropping query params/anchors
            clean = href.split("?")[0].split("#")[0]
            links.add(clean)

    print(f"  Found {len(links)} product links on this catalog page.")
    return links


def parse_duration(text: str) -> int:
    """
    Extract duration in minutes from free text, e.g., '40 minutes', '1 hour', '1-2 hours'.
    Returns 0 if unknown.
    """
    text = text.lower()

    # minutes
    m = re.search(r"(\d+)\s*min", text)
    if m:
        return int(m.group(1))

    # hours (simple)
    h = re.search(r"(\d+)\s*hour", text)
    if h:
        return int(h.group(1)) * 60

    # '1-2 hours' -> take the upper bound
    h_range = re.search(r"(\d+)\s*-\s*(\d+)\s*hour", text)
    if h_range:
        return int(h_range.group(2)) * 60

    return 0


def extract_test_types(soup: BeautifulSoup) -> List[str]:
    """
    Heuristically extract test type tags from buttons/badges or list items.
    This is tailored for SHL-style tags like:
    'Ability & Aptitude', 'Personality & Behavior', 'Assessment Exercises', etc.
    """
    tags = set()

    # Common pattern: badges or tag buttons
    for el in soup.select("a, span, div"):
        cls = " ".join(el.get("class") or []).lower()
        text = (el.get_text(" ", strip=True) or "").strip()
        if not text:
            continue
        if any(c in cls for c in ["tag", "badge", "chip", "pill"]):
            if len(text) < 60:  # avoid very long text
                tags.add(text)

    # Fallback: look for specific keywords in short elements
    KEYWORDS = [
        "ability", "aptitude", "numerical", "verbal", "logical", "reasoning",
        "personality", "behavior", "behaviour", "competencies", "simulation",
        "knowledge & skills", "assessment exercises", "development & 360",
        "biodata", "situational judgement", "coding", "programming",
    ]
    for el in soup.find_all(["li", "span"]):
        text = (el.get_text(" ", strip=True) or "").strip()
        if not text or len(text) > 60:
            continue
        low = text.lower()
        if any(kw in low for kw in KEYWORDS):
            tags.add(text)

    return sorted(tags)


def infer_flags_from_text(text: str) -> Dict[str, bool]:
    """
    Infer adaptive and remote flags from page text.
    """
    low = text.lower()
    adaptive = any(
        kw in low
        for kw in [
            "adaptive test",
            "computer adaptive",
            "cat test",
            "adaptive version",
        ]
    )
    remote = any(
        kw in low
        for kw in [
            "remote proctor",
            "remotely proctored",
            "online proctor",
            "remote test",
            "remote administration",
            "online assessment",
            "taken remotely",
        ]
    )
    return {"adaptive": adaptive, "remote": remote}


def scrape_assessment_page(driver: webdriver.Chrome, url: str) -> Assessment | None:
    """
    Visit a product detail page and extract assessment information.
    """
    try:
        print(f"  [Detail] {url}")
        driver.get(url)
        polite_sleep(4.0)
    except WebDriverException as e:
        print(f"    Failed to load {url}: {e}")
        return None

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Name: try h1 first, then fallback to <title>
    name_el = soup.find("h1")
    if name_el:
        name = name_el.get_text(" ", strip=True)
    else:
        title_el = soup.find("title")
        name = title_el.get_text(" ", strip=True) if title_el else url

    # Description: heuristic – first reasonably long <p> block
    description = ""
    for p in soup.find_all("p"):
        txt = (p.get_text(" ", strip=True) or "").strip()
        if len(txt) > 80:  # avoid tiny fragments
            description = txt
            break

    # Duration: search the entire text
    full_text = soup.get_text(" ", strip=True)
    duration = parse_duration(full_text)

    # Test types
    test_types = extract_test_types(soup)

    # Flags
    flags = infer_flags_from_text(full_text)

    return Assessment(
        name=name,
        url=url,
        description=description,
        duration_minutes=duration,
        adaptive=flags["adaptive"],
        remote=flags["remote"],
        test_types=test_types,
    )


def looks_like_individual_test(url: str) -> bool:
    """
    Heuristic filter to keep 'individual tests' and drop obvious bundles.
    To be conservative (to reach >=377 items), this filter is mild.
    """
    low = url.lower()

    # Exclude obvious job-solution / bundle style pages if needed.
    EXCLUDE_SUBSTRINGS = [
        "job-solution",
        "job-solution-",  # adjust if you know exact patterns
        "/solutions/for-",  # generic solution landing pages
    ]
    for s in EXCLUDE_SUBSTRINGS:
        if s in low:
            return False

    # Keep most catalog/view pages
    return "/product-catalog/view/" in low


def main():
    if not CATALOG_URLS:
        print(
            "ERROR: Please set CATALOG_URLS at the top of crawler_selenium.py "
            "to your SHL catalog listing URL(s) before running."
        )
        return

    driver = create_driver(headless=True)

    try:
        all_links: Set[str] = set()
        for cat_url in CATALOG_URLS:
            links = collect_product_links_from_catalog(driver, cat_url)
            all_links.update(links)

        print(f"\nTotal raw product links collected across catalogs: {len(all_links)}")

        # Filter to individual tests
        filtered_links = [u for u in all_links if looks_like_individual_test(u)]
        print(f"Links after 'individual test' filtering: {len(filtered_links)}")

        assessments: List[Assessment] = []
        seen_urls: Set[str] = set()

        for i, url in enumerate(sorted(filtered_links), start=1):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"\nScraping assessment {i}/{len(filtered_links)}")
            a = scrape_assessment_page(driver, url)
            if a is None:
                continue
            assessments.append(a)

            # Be gentle to the server
            polite_sleep(2.0)

        print(f"\nScraped {len(assessments)} assessments.")
        if len(assessments) < 377:
            print(
                f"WARNING: Only {len(assessments)} assessments scraped. "
                "Assignment expects at least 377 Individual Test Solutions. "
                "Consider adding more CATALOG_URLS or relaxing filters."
            )
        else:
            print("Requirement of >=377 Individual Test Solutions appears satisfied (subject to manual check).")

        # Save to JSON
        data = [asdict(a) for a in assessments]
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved assessments to {OUTPUT_PATH}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()