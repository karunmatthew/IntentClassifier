import json
import os
import copy

# CONSTANTS
WRITE = "w"
JSON = '.json'
PICK_AND_PLACE = "pick_and_place_simple"
OUT_FILE = "train_file_v2"
FOLDER_PATH = "/home/karun/Research_Project/alfred/data/json_feat_2.1.0/"
NO_OPERATION = 'NoOp'


# FOLDER_PATH = "/media/karun/My Passport/full_2.1.0/train/"


# returns a list of all json file paths within the provided folder path
def get_json_file_paths(folder_path):
    file_paths = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == JSON:
                file_path = os.path.join(path, name)
                file_paths.append(file_path)

    return file_paths


# returns true if the trial is of the passed task type
def is_of_task_type(json_object, filter_task_type):
    if filter_task_type == '':
        return True
    elif 'task_type' in json_object and \
            json_object['task_type'] == filter_task_type:
        return True
    else:
        return False


# parses the json file
def get_agent_data(json_object):
    init_action = json_object['scene']['init_action']
    agent_data = {'entityName': 'agent', 'relevant': 1,
                  'position': [init_action['x'], init_action['y'],
                               init_action['z'], 0, init_action['rotation'], 0]}
    return agent_data


# get the action sequence
def get_action_sequence(json_object):
    high_pddl = json_object['plan']['high_pddl']
    action_sequence = []
    for action in high_pddl:
        operation = action['discrete_action']['action']
        if not operation == NO_OPERATION:
            action_sequence.append(operation)
    return action_sequence


# returns the list of objects required for carrying out the task
def get_task_related_objects(json_object):
    related_objects = []
    high_pddl = json_object['plan']['high_pddl']
    for action in high_pddl:
        related_object, receptable_object = get_object_and_receptable(action)
        if not related_object is None and not related_object in related_objects:
            related_objects.append(related_object)
        if not receptable_object is None and not receptable_object in \
                                                 related_objects:
            related_objects.append(receptable_object)
    return related_objects


# extracts the object and receptable object if present from a high_pddl_action
def get_object_and_receptable(action):
    planner_action = action['planner_action']
    related_object = None
    receptable_object = None
    if 'objectId' in planner_action:
        related_object_data = planner_action['objectId'].split('|')
        related_object = {
            'entityName': related_object_data[0],
            'relevant': 1,
            'position': [float(related_object_data[1]),
                         float(related_object_data[2]),
                         float(related_object_data[3])]
        }

    if 'receptacleObjectId' in planner_action:
        related_object_data = planner_action['receptacleObjectId'].split(
            '|')
        receptable_object = {
            'entityName': related_object_data[0],
            'relevant': 1,
            'position': [float(related_object_data[1]),
                         float(related_object_data[2]),
                         float(related_object_data[3])]
        }

    return related_object, receptable_object


# when a high-idx is passed, it returns the corresponding
# high_pddl action with the same index
def get_corresponding_high_pddl_action(high_idx, json_object):
    high_pddl = json_object['plan']['high_pddl']
    return high_pddl[high_idx]


# given a list of all objects related to a task, it is required to find the
# subset of items relevant for a particular high_desc
def get_relevant_high_desc_objects(task_objects, json_object, high_idx,
                                   high_desc_command):
    # the list of all identified objects for the task
    high_desc_objects = copy.deepcopy(task_objects)
    high_pddl_action = get_corresponding_high_pddl_action(high_idx, json_object)
    related_object, receptable_object = get_object_and_receptable(
        high_pddl_action)
    action_args = []
    if 'args' in high_pddl_action['discrete_action']:
        action_args = high_pddl_action['discrete_action']['args']
    for high_desc_object in high_desc_objects:
        print(high_idx)
        print(high_desc_object['entityName'].lower())
        print(action_args)
        print('\n')
        if not high_desc_object['entityName'].lower() in \
               high_desc_command.lower() and \
                not high_desc_object['entityName'].lower() in action_args and \
                not high_desc_object == related_object and \
                not high_desc_object == receptable_object:
            high_desc_object['relevant'] = 0
    return high_desc_objects


def parse_json_file(file):
    task_descs_data = []
    high_descs_data = []

    with open(file) as json_file:
        json_object = json.load(json_file)
        agent_data = get_agent_data(json_object)
        language_annotations = json_object['turk_annotations']['anns']
        if is_of_task_type(json_object, PICK_AND_PLACE):
            for lang_ann in language_annotations:
                task_desc_data = {'task_desc': lang_ann["task_desc"].strip(),
                                  'high-idx': -1,
                                  'action_sequences': get_action_sequence(
                                      json_object)}
                task_related_objects = get_task_related_objects(json_object)
                task_desc_data['scene_description'] = [agent_data] + \
                                                      task_related_objects

                count = 0

                high_descs = lang_ann["high_descs"]
                for high_desc in high_descs:
                    high_desc_data = {
                        'high_idx': count,
                        'high_desc': high_desc.strip(),
                        'scene_description': [agent_data] +
                                             get_relevant_high_desc_objects(
                                                 task_related_objects,
                                                 json_object, count,
                                                 high_desc.strip())
                    }
                    high_descs_data.append(high_desc_data)
                    count = count + 1

                task_descs_data.append(task_desc_data)
        print(task_descs_data)
        print(high_descs_data)


outfile = open(OUT_FILE, WRITE)
# files = get_json_file_paths(FOLDER_PATH)
# for file_path in files:
#    parse_json_file(file_path)
parse_json_file('/home/karun/Research_Project/Docs/original_alfred[1].json')
outfile.close()
