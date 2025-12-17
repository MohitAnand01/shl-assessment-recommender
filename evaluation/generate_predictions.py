import csv
import requests
from typing import List

API_URL = "http://localhost:8000/recommend"
TEST_FILE = "test_data.csv"
OUTPUT_FILE = "predictions.csv"
K = 10  # number of recommendations per query


def load_test_queries(path: str) -> List[str]:
    """
    Load test queries from CSV.
    Assumes a column named 'Query' with the query text.
    Change 'Query' here if your column name is different.
    """
    queries = []
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if "Query" not in reader.fieldnames:
            raise ValueError(f"'Query' column not found in {path}. Columns: {reader.fieldnames}")
        for row in reader:
            q = row["Query"].strip()
            if q:
                queries.append(q)
    return queries


def call_api(query: str, k: int = 10) -> List[str]:
    """Call the /recommend API and return a list of assessment URLs."""
    payload = {"query": query, "k": k}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [item["url"] for item in data.get("recommended_assessments", [])]


def main():
    test_queries = load_test_queries(TEST_FILE)
    print(f"Loaded {len(test_queries)} test queries from {TEST_FILE}")

    rows = []
    for i, query in enumerate(test_queries, start=1):
        print(f"\nProcessing query {i}/{len(test_queries)}")
        print(f"Query: {query}")

        try:
            urls = call_api(query, k=K)
        except Exception as e:
            print(f"Error calling API: {e}")
            urls = []

        joined_urls = ",".join(urls)
        print(f"Recommended {len(urls)} URLs")

        rows.append({"query_id": i, "assessment_urls": joined_urls})

    # Write predictions.csv
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "assessment_urls"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()