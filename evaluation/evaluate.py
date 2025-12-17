import csv
import json
import requests
from typing import List, Dict

API_URL = "http://localhost:8000/recommend"
TRAIN_FILE = "train_data.csv"
K = 10  # Recall@10


def call_api(query: str, k: int = 10) -> List[str]:
    """Call the /recommend API and return a list of assessment URLs."""
    payload = {"query": query, "k": k}
    # API_URL already includes the full path
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [item["url"] for item in data.get("recommended_assessments", [])]

def recall_at_k(relevant: List[str], predicted: List[str], k: int = 10) -> float:
    """Compute Recall@K: |relevant ∩ predicted[:K]| / |relevant|."""
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = len(set(relevant) & set(top_k))
    return hits / len(relevant)


def load_train_data(path: str) -> List[Dict]:
    """
    Load train data from CSV with columns:
    - Query
    - Assessment_url (one relevant URL per row)
    
    Multiple rows can share the same Query. We group them into:
    {query: [url1, url2, ...]}
    """
    grouped: Dict[str, List[str]] = {}

    with open(path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        # Ensure expected headers are present
        if "Query" not in reader.fieldnames or "Assessment_url" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns 'Query' and 'Assessment_url', found: {reader.fieldnames}")

        for row in reader:
            q = row["Query"].strip()
            url = row["Assessment_url"].strip()
            if not q or not url:
                continue
            grouped.setdefault(q, []).append(url)

    records = []
    for q, urls in grouped.items():
        records.append({"query": q, "relevant_urls": urls})

    return records


def main():
    train_records = load_train_data(TRAIN_FILE)
    print(f"Loaded {len(train_records)} unique queries from {TRAIN_FILE}")

    results = []
    recalls = []

    for i, rec in enumerate(train_records, start=1):
        query = rec["query"]
        relevant_urls = rec["relevant_urls"]

        print(f"\n[{i}/{len(train_records)}] Query: {query}")
        print(f"Relevant URLs ({len(relevant_urls)}):")
        for u in relevant_urls:
            print(f"  - {u}")

        try:
            predicted_urls = call_api(query, k=K)
        except Exception as e:
            print(f"Error calling API for this query: {e}")
            predicted_urls = []

        r_at_k = recall_at_k(relevant_urls, predicted_urls, k=K)
        recalls.append(r_at_k)

        print(f"Recall@{K}: {r_at_k:.3f}")

        results.append(
            {
                "query": query,
                "relevant_urls": relevant_urls,
                "predicted_urls": predicted_urls,
                f"recall@{K}": r_at_k,
            }
        )

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\nMean Recall@{K}: {mean_recall:.3f}")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"k": K, "mean_recall": mean_recall, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Saved detailed results to evaluation_results.json")


if __name__ == "__main__":
    main()