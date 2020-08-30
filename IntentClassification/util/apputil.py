import os

JSON = '.json'


def get_json_file_paths(folder_path):
    file_paths = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == JSON:
                file_path = os.path.join(path, name)
                file_paths.append(file_path)

    return file_paths
