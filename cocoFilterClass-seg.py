import os
from shutil import copyfile
from pycocotools.coco import COCO
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from IPython.display import clear_output

IMAGE_FOLDER = "datasets/Offroad-Dataset-II-1/valid"
LABEL_FOLDER = "datasets/Offroad-Dataset-II-1/valid/_annotations.coco.json"
OUTPUT_FOLDER = "valid"


# Function to update and display the progress bar
def update_progress(progress):
    bar_length = 20  # Length of the progress bar
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text, end="\r")


# Function to output the data in TensorFlow CSV format
def tensorflowFormat(img_dict):
    total_progress = len(img_dict)
    progress = 0
    with open("train.csv", mode="w+", encoding="utf8") as f:
        # Write CSV header
        f.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

        for img_id in img_dict:
            progress += 1
            annotation_ids = coco.getAnnIds(img_id)
            annotations = coco.loadAnns(annotation_ids)

            image_meta = coco.loadImgs(annotations[0]["image_id"])[0]
            for ann in annotations:
                entity_id = ann["category_id"]
                entity = coco.loadCats(entity_id)[0]["name"]
                if entity in target_classes:
                    bbox = ann["bbox"]
                    f.write(
                        f"{image_meta['file_name']},{image_meta['width']},{image_meta['height']},"
                        f"{entity},{int(bbox[0])},{int(bbox[1])},{int(bbox[0] + bbox[2])},{int(bbox[1] + bbox[3])}\n"
                    )

            update_progress(progress / total_progress)


def yoloV8Format(img_dict):
    # category_mapping = {
    #     0: "person",
    #     1: "bicycle",
    #     2: "car",
    #     3: "motorcycle",
    #     5: "bus",
    #     7: "truck",
    #     8: "dog",
    #     9: "cat",

    # }
    category_mapping = {10: "rough_trail", 11: "puddle"}

    # Reverse the classes dictionary to get class ID from class name
    class_ids = {v: k for k, v in category_mapping.items()}

    total_progress = len(img_dict)
    progress = 0

    for img_id in img_dict:
        progress += 1
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)

        image_meta = coco.loadImgs(annotations[0]["image_id"])[0]

        # Copy the image file to the target folder
        source_image_path = os.path.join(COCO_IMAGES_DIRECTORY, image_meta["file_name"])
        target_image_path = os.path.join(
            EXTRACTED_SAVING_PATH, "images", image_meta["file_name"]
        )
        if not os.path.exists(os.path.join(EXTRACTED_SAVING_PATH, "images")):
            os.makedirs(os.path.join(EXTRACTED_SAVING_PATH, "images"))
        if not os.path.exists(target_image_path):
            copyfile(source_image_path, target_image_path)

        # Create the directory for the YOLO format if it doesn't exist
        if not os.path.exists(os.path.join(EXTRACTED_SAVING_PATH, "labels")):
            os.makedirs(os.path.join(EXTRACTED_SAVING_PATH, "labels"))
        filename = os.path.join(
            EXTRACTED_SAVING_PATH,
            "labels",
            image_meta["file_name"].replace("jpg", "txt"),
        )

        # Create the directory for the YOLO format if it doesn't exist
        with open(filename, "w+", encoding="utf8") as f:
            for ann in annotations:
                entity_id = ann["category_id"]
                entity = coco.loadCats(entity_id)[0]["name"]

                if entity in target_classes:
                    # Convert seg to YOLO format (normalized)
                    # Extract segmentation points
                    segmentation = ann["segmentation"][0]  # Assuming polygon format
                    x_coords = segmentation[0::2]
                    y_coords = segmentation[1::2]

                    # Calculate bounding box from segmentation
                    # Convert segmentation points to YOLO-seg format (normalized)
                    img_width = image_meta["width"]
                    img_height = image_meta["height"]
                    normalized_segmentation = [
                        f"{x / img_width:.6f} {y / img_height:.6f}"
                        for x, y in zip(x_coords, y_coords)
                    ]
                    segmentation_str = " ".join(normalized_segmentation)

                    class_id = class_ids[entity]

                    f.write(f"{class_id} {segmentation_str}\n")

        update_progress(progress / total_progress)


# Function to output the data in COCO XML format
def cocoFormat(img_dict):
    total_progress = len(img_dict)
    progress = 0

    for img_id in img_dict:
        progress += 1
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)

        image_meta = coco.loadImgs(annotations[0]["image_id"])[0]
        filename = os.path.join(
            EXTRACTED_SAVING_PATH, image_meta["file_name"].replace("jpg", "xml")
        )
        # Copy the image file to the target folder
        source_image_path = os.path.join(COCO_IMAGES_DIRECTORY, image_meta["file_name"])
        target_image_path = os.path.join(EXTRACTED_SAVING_PATH, image_meta["file_name"])
        if not os.path.exists(target_image_path):
            copyfile(source_image_path, target_image_path)

        # Create the directory for the COCO XML format if it doesn't exist
        with open(filename, "w+", encoding="utf8") as f:
            annotation = Element("annotation")

            folder = SubElement(annotation, "folder")
            folder.text = SAVE_FOLDER

            filename_elem = SubElement(annotation, "filename")
            filename_elem.text = image_meta["file_name"]

            path = SubElement(annotation, "path")
            path.text = os.path.join(EXTRACTED_SAVING_PATH, image_meta["file_name"])

            source = SubElement(annotation, "source")
            database = SubElement(source, "database")
            database.text = "Unknown"

            size = SubElement(annotation, "size")
            width = SubElement(size, "width")
            width.text = str(image_meta["width"])
            height = SubElement(size, "height")
            height.text = str(image_meta["height"])
            depth = SubElement(size, "depth")
            depth.text = "3"

            segmented = SubElement(annotation, "segmented")
            segmented.text = "0"

            for ann in annotations:
                entity_id = ann["category_id"]
                entity = coco.loadCats(entity_id)[0]["name"]

                if entity in target_classes:
                    obj = SubElement(annotation, "object")

                    name = SubElement(obj, "name")
                    name.text = entity

                    pose = SubElement(obj, "pose")
                    pose.text = "Unspecified"

                    truncated = SubElement(obj, "truncated")
                    truncated.text = "0"

                    difficult = SubElement(obj, "difficult")
                    difficult.text = "0"

                    bndbox = SubElement(obj, "bndbox")
                    bbox = ann["bbox"]
                    xmin = SubElement(bndbox, "xmin")
                    xmin.text = str(int(bbox[0]))
                    ymin = SubElement(bndbox, "ymin")
                    ymin.text = str(int(bbox[1]))
                    xmax = SubElement(bndbox, "xmax")
                    xmax.text = str(int(bbox[0] + bbox[2]))
                    ymax = SubElement(bndbox, "ymax")
                    ymax.text = str(int(bbox[1] + bbox[3]))

            # Write XML to file
            f.write(parseString(tostring(annotation, pretty_print=True)).toprettyxml())

        update_progress(progress / total_progress)


# Get the current working directory
current_path = os.path.abspath(os.getcwd())

# Define paths for COCO annotations, images, and extracted dataset
COCO_ANNOTATIONS_PATH = os.path.join(current_path, LABEL_FOLDER)
COCO_IMAGES_DIRECTORY = os.path.join(current_path, IMAGE_FOLDER)
EXTRACTED_SAVING_PATH = os.path.join(current_path, OUTPUT_FOLDER)
SAVE_FOLDER = os.path.basename(os.path.dirname(EXTRACTED_SAVING_PATH))

# Create the directory for extracted dataset if it doesn't exist
if not os.path.exists(EXTRACTED_SAVING_PATH):
    os.mkdir(EXTRACTED_SAVING_PATH)

# Load COCO annotations
coco = COCO(COCO_ANNOTATIONS_PATH)

# Load all categories
cats = coco.loadCats(coco.getCatIds())
nms = [cat["name"] for cat in cats]

# Read target classes from the classes.txt file
target_classes = []
with open(os.path.join(current_path, "classes.txt"), "r") as f:
    for line in f.readlines():
        target_classes.append(line.strip())

# Create a dictionary to store image IDs and their corresponding classes
img_dict = {}
for classes in target_classes:
    catIds = coco.getCatIds(catNms=[classes])
    imgIds = coco.getImgIds(catIds=catIds)
    for imgID in imgIds:
        content = img_dict.get(imgID, "")
        if content:
            content += ","
        img_dict[imgID] = content + classes


# Run the desired output format function
# tensorflowFormat(img_dict)
yoloV8Format(img_dict)
# cocoFormat(img_dict)
