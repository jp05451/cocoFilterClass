import os
from shutil import copyfile
from pycocotools.coco import COCO
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from IPython.display import clear_output

IMAGE_FOLDER="train2017"
LABEL_FOLDER="annotations/instances_train2017.json"

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
    print(text,end="\r")

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


# Function to output the data in COCO XML format
def cocoFormat(img_dict):
    total_progress = len(img_dict)
    progress = 0
    for img_id in img_dict:
        progress += 1
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)

        image_meta = coco.loadImgs(annotations[0]["image_id"])[0]
        node_root = Element("annotation")
        node_folder = SubElement(node_root, "folder")
        node_folder.text = SAVE_FOLDER
        node_filename = SubElement(node_root, "filename")
        node_filename.text = image_meta["file_name"]
        node_size = SubElement(node_root, "size")
        node_width = SubElement(node_size, "width")
        node_width.text = str(image_meta["width"])
        node_height = SubElement(node_size, "height")
        node_height.text = str(image_meta["height"])
        node_depth = SubElement(node_size, "depth")
        node_depth.text = "3"

        for ann in annotations:
            entity_id = ann["category_id"]
            entity = coco.loadCats(entity_id)[0]["name"]
            if entity in target_classes:
                node_object = SubElement(node_root, "object")
                node_name = SubElement(node_object, "name")
                node_name.text = entity
                node_difficult = SubElement(node_object, "difficult")
                node_difficult.text = "0"
                node_bndbox = SubElement(node_object, "bndbox")
                bbox = ann["bbox"]
                node_xmin = SubElement(node_bndbox, "xmin")
                node_xmin.text = str(round(bbox[0]))
                node_ymin = SubElement(node_bndbox, "ymin")
                node_ymin.text = str(round(bbox[1]))
                node_xmax = SubElement(node_bndbox, "xmax")
                node_xmax.text = str(round(bbox[0] + bbox[2]))
                node_ymax = SubElement(node_bndbox, "ymax")
                node_ymax.text = str(round(bbox[1] + bbox[3]))

        xml = tostring(node_root, pretty_print=True)
        dom = parseString(xml)
        with open(
            os.path.join(
                EXTRACTED_SAVING_PATH, image_meta["file_name"].split(".")[0] + ".xml"
            ),
            "w",
        ) as xml_file:
            xml_file.write(dom.toxml())

        copyfile(
            os.path.join(COCO_IMAGES_DIRECTORY, image_meta["file_name"]),
            os.path.join(EXTRACTED_SAVING_PATH, image_meta["file_name"]),
        )

        update_progress(progress / total_progress)


# Get the current working directory
current_path = os.path.abspath(os.getcwd())

# Define paths for COCO annotations, images, and extracted dataset
COCO_ANNOTATIONS_PATH = os.path.join(
    current_path,LABEL_FOLDER
)
COCO_IMAGES_DIRECTORY = os.path.join(current_path,IMAGE_FOLDER)
EXTRACTED_SAVING_PATH = os.path.join(current_path, "extracted_dataset")
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
tensorflowFormat(img_dict)
# cocoFormat(img_dict)
