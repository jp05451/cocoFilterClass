"""
datasets/
└── ebike2.0-7/
    └── valid/
        ├── images/
        ├── labels_seg/    <-- segmentation 格式原始標籤
        └── labels_bbox/   <-- ✅ 轉換後 bbox 格式會放這裡
"""

import os
from numba import jit

@jit
def polygon_to_bbox(points):
    xs = points[::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

@jit
def isPolygon(points):
    if len(points) == 4:
        return False
    return True


def convert_yolo_seg_to_bbox(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 3:
                    continue  # skip invalid lines

                class_id = int(parts[0])
                coords = parts[1:]

                if not isPolygon(coords):
                    x_center, y_center, width, height = coords
                else:
                    # segmentation -> bbox
                    x_center, y_center, width, height = polygon_to_bbox(coords)
                f_out.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )
    
    print(f"✅ 已完成轉換，輸出到：{output_dir}")


# ✅ 使用方式：
# 修改這兩個路徑即可
data_folder = "datasets/ebike2.0-7"
# input_folder = "datasets/ebike2.0-7/valid/labels"
output_folder = "datasets/ebike2.0-7/valid/labels_bbox"

convert_yolo_seg_to_bbox(
    os.path.join(data_folder, "train/labels"), os.path.join(data_folder, "train/labels_bbox")
)
convert_yolo_seg_to_bbox(
    os.path.join(data_folder, "valid/labels"), os.path.join(data_folder, "valid/labels_bbox")
)
convert_yolo_seg_to_bbox(
    os.path.join(data_folder, "test/labels"), os.path.join(data_folder, "test/labels_bbox")
)
