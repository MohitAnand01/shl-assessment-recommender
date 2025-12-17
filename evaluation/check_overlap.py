import json
import csv
import os

ASSESSMENTS_PATH = os.path.join("..", "backend", "data", "assessments.json")
TRAIN_PATH = "train_data.csv"

# Load crawled assessments
with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
    assessments = json.load(f)

crawled_urls = set(a["url"] for a in assessments)

# Load train labels (use a Windows-friendly encoding)
with open(TRAIN_PATH, "r", encoding="latin-1", newline="") as f:
    reader = csv.DictReader(f)
    train_urls = set(row["Assessment_url"] for row in reader)

overlap = crawled_urls.intersection(train_urls)

print(f"Crawled URLs: {len(crawled_urls)}")
print(f"Train label URLs: {len(train_urls)}")
print(f"Overlap: {len(overlap)}")

if overlap:
    print("\nSome matching URLs:")
    for url in list(overlap)[:10]:
        print(f"  - {url}")
else:
    print("\nNo overlap found.")
    print("\nExample crawled URL:")
    print(f"  {next(iter(crawled_urls))}")
    print("\nExample train label URL:")
    print(f"  {next(iter(train_urls))}")