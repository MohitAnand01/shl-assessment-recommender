import json

path = "data/assessments.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

patched_test_type = 0
patched_duration = 0
patched_adaptive = 0
patched_remote = 0

for item in data:
    # 1) test_type <- test_types
    if "test_type" not in item and "test_types" in item:
        item["test_type"] = item["test_types"]
        patched_test_type += 1

    # 2) duration <- duration_minutes
    if "duration" not in item and "duration_minutes" in item:
        item["duration"] = item["duration_minutes"]
        patched_duration += 1

    # 3) adaptive_support <- adaptive (Yes/No)
    if "adaptive_support" not in item and "adaptive" in item:
        item["adaptive_support"] = "Yes" if item["adaptive"] else "No"
        patched_adaptive += 1

    # 4) remote_support <- remote (Yes/No)
    if "remote_support" not in item and "remote" in item:
        item["remote_support"] = "Yes" if item["remote"] else "No"
        patched_remote += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Patched test_type for {patched_test_type} items.")
print(f"Patched duration for {patched_duration} items.")
print(f"Patched adaptive_support for {patched_adaptive} items.")
print(f"Patched remote_support for {patched_remote} items.")
print(f"Total assessments: {len(data)}")