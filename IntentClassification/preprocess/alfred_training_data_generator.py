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
GOTO_LOCATION = 'GotoLocation'
PICKUP_ACTION = 'PickupObject'
PUTDOWN_ACTION = 'PutObject'

agent_data = {}
agent_init_data = {}


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
    return True
    if filter_task_type == '':
        return True
    elif 'task_type' in json_object and \
            json_object['task_type'] == filter_task_type:
        return True
    else:
        return False


# parses the json file
def init_agent_data(json_object):
    init_action = json_object['scene']['init_action']
    global agent_data, agent_init_data
    agent_data = {'entityName': 'agent', 'relevant': 1,
                  'position': [init_action['x'], init_action['y'],
                               init_action['z'], 0, init_action['rotation'], 0]}
    agent_init_data = {'entityName': 'agent', 'relevant': 1,
                       'position': [init_action['x'], init_action['y'],
                                    init_action['z'], 0,
                                    init_action['rotation'], 0]}


def update_agent_data(location_string):
    loc = location_string.split('|')
    agent_data['position'] = [float(loc[1]), float(loc[2]), float(loc[3]), 0.0,
                              0.0, 0.0]


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
        # we mark an object as irrelevant if its not mentioned in high_desc
        # if its not present in args property of pddl_action
        # if its not present in objectId and ReceptableObjectId
        #
        # The below line should be added if we look inside the language
        # not high_desc_object['entityName'].lower()
        # in high_desc_command.lower()
        #
        if not high_desc_object['entityName'].lower() in action_args and \
                not high_desc_object == related_object and \
                not high_desc_object == receptable_object:
            high_desc_object['relevant'] = 0
    return high_desc_objects


def deduplicate_scene_description(scene_description):
    agent_state = []

    # for scene

    return scene_description


def merge_scene_descriptions(scene_description, scene):

    match = False
    for a_scene in scene_description:
        if a_scene['entityName'] == scene['entityName'] \
                and a_scene['position'] == scene['position']:
            a_scene['relevant']  = a_scene['relevant'] or scene['relevant']
            match = True

    if not match:
        scene_description.append(scene)


def get_merged_high_desc(high_descs):
    multi_desc = {}
    high_idxs = []
    task_desc = []
    scene_description = []
    action_sequence = []
    for high_desc in high_descs:
        high_idxs.append(high_desc['high_idx'][0])
        task_desc.append(high_desc['high_desc'])
        action_sequence.append(high_desc['action_sequence'][0])
        for scene in high_desc['scene_description']:
            if not scene in scene_description:
                merge_scene_descriptions(scene_description, scene)
    multi_desc['high_idx'] = high_idxs
    multi_desc['high_descs'] = task_desc
    multi_desc['action_sequence'] = action_sequence
    multi_desc['scene_description'] = deduplicate_scene_description(
        scene_description)
    return multi_desc


def get_multi_high_descs(task_high_desc):
    multi_high_desc = []
    for i in range(len(task_high_desc)):
        temp = []
        for j in range(i+1, len(task_high_desc)):
            temp.append(task_high_desc[j])
            multi_high_desc.append(get_merged_high_desc(temp))

    return multi_high_desc


def parse_json_file(file):
    task_descs_data = []
    high_descs_data = []
    multi_descs_data = []

    with open(file) as json_file:
        json_object = json.load(json_file)
        language_annotations = json_object['turk_annotations']['anns']
        task_id = json_object['task_id']
        if is_of_task_type(json_object, PICK_AND_PLACE):
            for lang_ann in language_annotations:
                task_high_desc = []
                # initialize the agent location at the start of every task
                init_agent_data(json_object)
                action_sequences = get_action_sequence(json_object)
                task_desc_data = {'task_desc': lang_ann["task_desc"].strip(),
                                  'high-idx': -1,
                                  'task_id': task_id,
                                  'action_sequence': action_sequences}
                task_related_objects = get_task_related_objects(json_object)
                task_desc_data['scene_description'] = [copy.deepcopy(
                    agent_data)] + task_related_objects

                count = 0

                high_descs = lang_ann["high_descs"]
                assignment_id = lang_ann["assignment_id"]
                for high_desc in high_descs:
                    # Update agents location after the high_desc is created
                    if action_sequences[count] == PICKUP_ACTION:
                        update_agent_data(json_object['plan']['high_pddl']
                                          [count]['planner_action']['objectId'])
                    elif action_sequences[count] == PUTDOWN_ACTION:
                        update_agent_data(json_object['plan']['high_pddl']
                                          [count]['planner_action']
                                          ['receptacleObjectId'])

                    high_desc_data = {
                        'high_idx': [count],
                        'assignment_id': assignment_id,
                        'high_desc': high_desc.strip(),
                        'action_sequence': [action_sequences[count]],
                        'scene_description': [copy.deepcopy(agent_data)] +
                                             get_relevant_high_desc_objects(
                                                 task_related_objects,
                                                 json_object, count,
                                                 high_desc.strip())
                    }
                    task_high_desc.append(high_desc_data)
                    high_descs_data.append(high_desc_data)
                    count = count + 1
                multi_descs_data.append(get_multi_high_descs(task_high_desc))
                task_descs_data.append(task_desc_data)
        print({'tasks': task_descs_data})
        print({'high_descs': high_descs_data})
        print({'multi_high_descs': multi_descs_data})


outfile = open(OUT_FILE, WRITE)
# files = get_json_file_paths(FOLDER_PATH)
# for file_path in files:
#    parse_json_file(file_path)
# parse_json_file('/home/karun/Research_Project/Docs/original_alfred[1].json')
parse_json_file('/home/karun/Research_Project/alfred/data/json_feat_2.1.0'
                '/pick_and_place_simple-Statue-None-CoffeeTable-228'
                '/trial_T20190906_185451_580211/pp/ann_0.json')
# parse_json_file('/home/karun/Research_Project/alfred/data/json_feat_2.1.0'
#                '/pick_and_place_with_movable_recep-TissueBox-Plate'
#                '-DiningTable-203/trial_T20190909_134437_433211/pp/ann_0.json')
outfile.close()
