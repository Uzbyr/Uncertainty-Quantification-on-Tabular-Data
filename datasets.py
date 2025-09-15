import os
import json
from urllib.parse import urlparse, parse_qs

# Root folder containing all the subfolders
root_folder = "data/"

results = []

for subdir, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".json"):  # only look at JSON files
            filepath = os.path.join(subdir, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                folder_name = os.path.basename(subdir)  # use folder name
                task_type = data.get("task_type", "N/A")
                if task_type == "regression":
                    continue

                source_url = data.get("source", "")
                parsed = urlparse(source_url)
                query = parse_qs(parsed.query)
                id = query.get("id", ["N/A"])[0]

                if id == "N/A":
                    continue

                n_num_features = data.get("n_num_features", "N/A")
                n_cat_features = data.get("n_cat_features", "N/A")
                train_size = data.get("train_size", 0)  # numeric for sorting
                val_size = data.get("val_size", "N/A")
                test_size = data.get("test_size", "N/A")


                results.append((
                    folder_name,
                    task_type,
                    n_num_features,
                    n_cat_features,
                    train_size,
                    val_size,
                    test_size,
                    id
                ))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Sort results by train_size (descending for largest first)
results.sort(key=lambda x: x[4], reverse=True)

# Print header
# print(f"{'Folder':25} {'Task':15} {'#Num/#Cat':15} {'Train/Val/Test':20} {'ID':10}")
# print("-" * 95)

# # Print rows
# for folder_name, task_type, n_num, n_cat, train_size, val_size, test_size, id in results:
#     print(f"{folder_name[:25]:25} {task_type:15} {f'{n_num}/{n_cat}':15} {f'{train_size}/{val_size}/{test_size}':20} {id:10}")

print("[", end="")
for folder_name, task_type, n_num, n_cat, train_size, val_size, test_size, id in results:
      if train_size <= 10_000:
          print(id, end=",")
print("]")

