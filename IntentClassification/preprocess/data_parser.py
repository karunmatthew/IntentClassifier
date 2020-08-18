# This program contains utility functions to help with the
# pre-processing of the training data. As we focus on only
# kitchen-floor-plans and simple pick and place experiments
# we parse the JSON file in the alfred dataset and copy the
# ones that satisfy the condition to the /data-input folder
# and also generate an OUT_FILE which contains all the high
# descriptions of the trials

import json
import os
from shutil import copy
import ntpath

# CONSTANTS
WRITE_OPTION = "w"
INDEX = 1

# ----------------------------------------
# CONFIGURATION VALUES ARE SPECIFIED BELOW
# ----------------------------------------
# FOLDER_PATH= "/home/karun/Research_Project/alfred/data/json_2.1.0/tests_seen/"
FOLDER_PATH = "/home/karun/Research_Project/alfred/data/json_feat_2.1.0/"
# FOLDER_PATH = "/media/karun/My Passport/full_2.1.0/train/"
# file to write out the high description texts
OUT_FILE = "outfile_pickup_simple"
# the type of task trials to consider
# if task_type is '', then all types of
# tasks are considered
task_type = "pick_and_place_simple"
# -----------------------------------------

outfile = open(OUT_FILE, WRITE_OPTION)

max_y = -100
min_y = 9999


# extracts the high desc and task desc for each of the json file passed
def parse_json_file(file, filter_task_type):
    # print(file)
    with open(file) as json_file:
        json_object = json.load(json_file)
        language_annotations = json_object['turk_annotations']['anns']
        if 0 and is_kitchen_floor_plan(json_object) and \
                is_of_task_type(json_object, filter_task_type):
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
        else:
            get_agent_height(json_object)

# returns true if the floor plan is of type kitchen
# FloorPlan1 - FloorPlan30 is kitchen plan
def is_kitchen_floor_plan(json_object):
    floor_plan = json_object['scene']['floor_plan']
    plan_number = int(floor_plan.split("FloorPlan")[1])
    if 1 <= plan_number <= 30:
        return True
    else:
        return False


# returns the y-coordinate of the agent
def get_agent_height(json_object):
    y = json_object['scene']['init_action']['y']
    x = json_object['scene']['init_action']['x']
    z = json_object['scene']['init_action']['horizon']
    global max_y
    global min_y
    print(x, "  ", y, "  ", z)
    if y > max_y:
        max_y = y
    if y < min_y:
        min_y = y
    return y


# returns true if the trial is of the passed task type
def is_of_task_type(json_object, filter_task_type):
    if filter_task_type == '':
        return True
    elif 'task_type' in json_object and json_object['task_type'] == filter_task_type:
        return True
    else:
        return False


# copies all kitchen floor plan json files into the data folder of the project
def process_files(folder_path):
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == '.json':
                file_path = os.path.join(path, name)
                parse_json_file(file_path, task_type)


process_files(FOLDER_PATH)
outfile.close()

# TESTING CODE
# FILE_PATH = "/home/karun/Research_Project/alfred/data/json_2.1.0/tests_seen/" \
#             "trial_T20190907_032644_545223/traj_data.json"
# parse_json_file(FILE_PATH)