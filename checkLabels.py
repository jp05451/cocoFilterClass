import os
import yaml
from YAMLtools import getYamlFolder, getYamlPath


def check_and_fix_labels(label_file):
    with open(label_file, "r") as file:
        lines = file.readlines()

    fixed_lines = []
    seen_labels = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            label = parts[0]
            coords = tuple(map(float, parts[1:]))
            if (label, coords) not in seen_labels:
                seen_labels.add((label, coords))
                if all(0 <= coord <= 1 for coord in coords):
                    fixed_lines.append(line)
                else:
                    print(
                        f"Found out of bounds coordinates in {label_file}: {line.strip()}"
                    )
                    # Fix the coordinates here if possible
                    # For example, you can clip the values to be within [0, 1]
                    fixed_coords = [max(0, min(1, coord)) for coord in coords]
                    fixed_line = (
                        f"{parts[0]} " + " ".join(map(str, fixed_coords)) + "\n"
                    )
                    fixed_lines.append(fixed_line)
            else:
                print(f"Duplicate label found in {label_file}: {line.strip()}")

    # with open(label_file, "w") as file:
    #     file.writelines(fixed_lines)


# def get_data_folders(yaml_file):
#     with open(yaml_file, "r") as file:
#         data = yaml.safe_load(file)
#     folders = []
#     for key in ["train", "val", "test"]:
#         if key in data:
#             folders.extend(data[key])
#     return folders


yaml_file = "/data/ebike/datasets/multiDatasets.yaml"
data_folders = getYamlFolder(yaml_file)
data_path = getYamlPath(yaml_file)

for folder in data_folders:
    label_dir = os.path.join(data_path, folder, "labels")
    if os.path.exists(label_dir):
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith(".txt"):
                    check_and_fix_labels(os.path.join(root, file))
