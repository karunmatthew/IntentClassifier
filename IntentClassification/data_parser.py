import json
import os
from shutil import copy
import ntpath

FOLDER_PATH = "/home/karun/Research_Project/alfred/data/json_2.1.0/tests_seen/"
FILE_PATH = "/home/karun/Research_Project/alfred/data/json_2.1.0/tests_seen/" \
            "trial_T20190907_032644_545223/traj_data.json"
WRITE_OPTION = "w"
OUT_FILE = "outfile"

INDEX = 1
outfile = open(OUT_FILE, WRITE_OPTION)


# extracts the high desc and task desc for each of the json file passed
def parse_json_file(file):

    with open(file) as json_file:
        json_object = json.load(json_file)
        language_annotations = json_object['turk_annotations']['anns']
        if is_kitchen_floor_plan(json_object):
            global INDEX
            print("\n" + file)
            copy(file, "./data-input/" + os.path.splitext(ntpath.basename(file))[0] + str(INDEX) + ".json")
            INDEX = INDEX + 1
            for lang_ann in language_annotations:
                high_descs = lang_ann["high_descs"]
                task_desc = lang_ann["task_desc"]
                for task in high_descs:
                    outfile.write(task + "\n")
                print(task_desc, "    ", high_descs)


# returns true if the floor plan is of type kitchen
# FloorPlan1 - FloorPlan30 is kitchen plan
def is_kitchen_floor_plan(json_object):
    floor_plan = json_object['scene']['floor_plan']
    plan_number = int(floor_plan.split("FloorPlan")[1])
    if 1 <= plan_number <= 30:
        return True
    else:
        return False


# copies all kitchen floor plan json files into the data folder of the project
def process_files(folder_path):
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            file_path = os.path.join(path, name)
            parse_json_file(file_path)



#parse_json_file(FILE_PATH)

process_files(FOLDER_PATH)
outfile.close()
