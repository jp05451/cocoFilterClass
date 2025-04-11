import yaml


def getYamlFolder(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    folders = []
    for key in ["train", "val", "test"]:
        if key in data:
            folders.extend(data[key])
    return folders


def getYamlPath(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data["path"]
