from shapely.geometry import Polygon, box
import numpy as np
from tools import find_files
from numba import jit
import os


dataset_paths = "valid"
output_paths = "valid_bbox"

def grid_cutting(polygon_coords, grid_size):
    """
    polygon_coords: List of (x, y) tuples
    grid_size: length of each square side
    returns: list of (x1, y1, x2, y2) of grid squares that intersect with polygon
    """
    poly = Polygon(polygon_coords)
    minx, miny, maxx, maxy = poly.bounds

    grid_boxes = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(
                x, y, x + grid_size, y + grid_size
            )  # box(left, bottom, right, top)
            if poly.intersects(cell):
                grid_boxes.append((x, y, x + grid_size, y + grid_size))
            y += grid_size
        x += grid_size

    return grid_boxes

@jit(cache=True)
def convert_to_yolo(grid_box, class_id):
    """
    Convert grid box coordinates to YOLO format
    grid_box: (x1, y1, x2, y2)
    returns: (class_id, x_center, y_center, width, height)
    """
    x1, y1, x2, y2 = grid_box
    x_center = round((x1 + x2) / 2, 6)
    y_center = round((y1 + y2) / 2, 6)
    width = round(x2 - x1, 6)
    height = round(y2 - y1, 6)
    return class_id, x_center, y_center, width, height


def main():
    files = find_files(dataset_paths, ".txt")
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            # convert part to float and zip them into tuples
            coords = list(zip(map(float, parts[1::2]), map(float, parts[2::2])))

            # get grid boxes
            grid_boxes = grid_cutting(coords, 0.05)
            for grid_box in grid_boxes:
                yolo_box = convert_to_yolo(grid_box, class_id)
                new_lines.append(" ".join(map(str, yolo_box)) + "\n")
    
        # write new lines to file
        new_file = file.replace(dataset_paths, output_paths)
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "w") as f:
            f.writelines(new_lines)
            
    print(f"✅ 已完成轉換，輸出到：{output_paths}")
if __name__ == "__main__":
    main()