import os


def find_files(directory, subTitle=""):
    returnFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(subTitle):
                returnFiles.append(os.path.join(root, file))
                # print(file)
    return returnFiles
