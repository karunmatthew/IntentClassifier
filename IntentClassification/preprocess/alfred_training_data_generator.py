import json
import copy
import sys
from util.apputil import get_json_file_paths
from util.alfred_json_parser import get_action_sequence, \
    get_object_and_receptable, is_of_task_type, get_task_related_objects, \
    get_floor_plan

# CONSTANTS
APPEND = "a"
JSON = '.json'
PICK_AND_PLACE = "pick_and_place_simple"
TRAINING_SET_FILE = "../data-train/training_set.txt"
TESTING_SET_FILE = "../data-test/testing_set.txt"
FOLDER_PATH = "/home/karun/Research_Project/alfred/data/json_feat_2.1.0/"
# FOLDER_PATH = "/media/karun/My Passport/full_2.1.0/train/"

GOTO_LOCATION = 'GotoLocation'
PICKUP_ACTION = 'PickupObject'
PUTDOWN_ACTION = 'PutObject'

# We need to divide the data set into train and test sets
# We also limit training to a small subset of floor plans
# to see if the system is able to generalize to different
# types of environments
train_floor_plans = []
test_floor_plans = []
trial_count = 0
TRAIN_PERCENT = 0.10
floor_plans = {}

agent_data = {}
agent_init_data = {}

# TODO REMOVE
# Collecting statistics
a_seq_total = 0.0
a_seq_count = 0.0
files_total = 0.0
task_desc_total = 0.0
multi_desc_count = 0.0
high_desc_count = 0.0


# FOLDER_PATH = "/media/karun/My Passport/full_2.1.0/train/"

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


# when a high-idx is passed, it returns the corresponding
# high_pddl action with the same index
def get_corresponding_high_pddl_action(high_idx, json_object):
    high_pddl = json_object['plan']['high_pddl']
    return high_pddl[high_idx]


# given a list of all objects related to a task, it is required to find the
# subset of items relevant for a particular high_desc
def get_relevant_high_desc_objects(task_objects, json_object, high_idx):
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


# when creating multi-descs we merge scene descriptions of each of the high-desc
# an object is made relevant if it is relevant in any of the high descs
def merge_scene_descriptions(scene_description, scene):
    match = False
    for a_scene in scene_description:
        if a_scene['entityName'] == scene['entityName'] \
                and a_scene['position'] == scene['position']:
            # this object has already been added, so no need to add again
            match = True
            # if the object has become relevant, make it relevant
            a_scene['relevant'] = a_scene['relevant'] or scene['relevant']
        if a_scene['entityName'] == scene['entityName'] \
                and a_scene['entityName'] == 'agent':
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
        task_desc = task_desc + high_desc['desc']
        action_sequence.append(high_desc['action_sequence'][0])
        for scene in high_desc['scene_description']:
            if scene not in scene_description:
                merge_scene_descriptions(scene_description, scene)
    multi_desc['record_type'] = 'multi_desc'
    multi_desc['desc'] = task_desc
    multi_desc['high_idx'] = high_idxs
    multi_desc['assignment_id'] = high_descs[0]['assignment_id']
    multi_desc['action_sequence'] = action_sequence
    multi_desc['scene_description'] = scene_description
    return multi_desc


# creates the multi-desc records from the high-desc list
# if a list of 4 high-descs are passed
# then the generated multi-desc would be
# [0,1] [0,1,2] [0,1,2,3] [1,2] [1,2,3] [2,3]
def get_multi_high_descs(task_high_desc):
    multi_high_desc = []
    for i in range(len(task_high_desc)):
        for j in range(i + 1, len(task_high_desc)):
            # choose different combinations of high-desc in order
            temp = task_high_desc[i: j + 1]
            multi_high_desc.append(copy.deepcopy(get_merged_high_desc(temp)))

    return multi_high_desc


def parse_json_file(file):
    task_descs_data = []
    high_descs_data = []
    multi_descs_data = []

    is_training_record = False

    with open(file) as json_file:

        json_object = json.load(json_file)
        language_annotations = json_object['turk_annotations']['anns']
        task_id = json_object['task_id']
        is_slice_task = json_object['pddl_params']['object_sliced']

        if is_of_task_type(json_object, PICK_AND_PLACE) and not is_slice_task:
            # print(file)
            # identify the floor plan and decide whether this file data
            # is to be added to training set or testing set
            floor_plan = get_floor_plan(json_object)
            if floor_plan in train_floor_plans:
                is_training_record = True

            # iterate through each language annotation of different Turks
            for lang_ann in language_annotations:
                task_high_desc = []
                # initialize the agent location at the start of every task
                init_agent_data(json_object)
                action_sequences = get_action_sequence(json_object)

                # --------------------- TASK-DESC ------------------------ #
                task_desc_data = {'record_type': 'task_desc',
                                  'desc': [lang_ann["task_desc"].strip()],
                                  'high_idx': [-1],
                                  'task_id': task_id,
                                  'action_sequence': action_sequences}

                task_related_objects = get_task_related_objects(json_object)

                task_desc_data['scene_description'] = [copy.deepcopy(
                    agent_data)] + task_related_objects
                task_descs_data.append(task_desc_data)
                # -------------------------------------------------------- #

                # -------------------- Object Close to Receptable -------- #
                if len(action_sequences) == 4 and \
                        action_sequences[1] == 'PickupObject' and \
                        action_sequences[2] == 'GotoLocation' and \
                        action_sequences[3] == 'PutObject':
                    task_desc_data_GPP = {'record_type': 'task_desc',
                                          'desc': [lang_ann["task_desc"].strip()],
                                          'high_idx': [-2],
                                          'task_id': task_id,
                                          'action_sequence': ['GotoLocation',
                                                              'PickupObject',
                                                              'PutObject']}
                    GPP_related_objects = copy.deepcopy(get_task_related_objects(json_object))
                    # update position of receptable as position of object
                    GPP_related_objects[1]['position'] = GPP_related_objects[0]['position']
                    task_desc_data_GPP['scene_description'] = [copy.deepcopy(
                        agent_data)] + GPP_related_objects
                    task_descs_data.append(task_desc_data_GPP)
                # -------------------------------------------------------- #

                # -------------------- Agent next to object -------------- #
                if len(action_sequences) == 4 and \
                        action_sequences[1] == 'PickupObject' and \
                        action_sequences[0] == 'GotoLocation':
                    task_desc_data_PGP = {'record_type': 'task_desc',
                                          'desc': [lang_ann["task_desc"].strip()],
                                          'high_idx': [-3],
                                          'task_id': task_id,
                                          'action_sequence': ['PickupObject',
                                                              'GotoLocation',
                                                              'PutObject']}
                    PGP_related_objects = copy.deepcopy(get_task_related_objects(json_object))
                    # update position of agent as that of picked object
                    PGP_agent_data = copy.deepcopy(agent_data)
                    PGP_agent_data['position'] = copy.deepcopy(PGP_related_objects[0]['position'])
                    task_desc_data_PGP['scene_description'] = [PGP_agent_data] + PGP_related_objects
                    task_descs_data.append(task_desc_data_PGP)
                # ----------------------------------------------------------------------- #

                # -------------------- Agent next to object and receptable -------------- #
                if len(action_sequences) == 4 and \
                        action_sequences[1] == 'PickupObject' and \
                        action_sequences[2] == 'GotoLocation' and \
                        action_sequences[3] == 'PutObject' and \
                        action_sequences[0] == 'GotoLocation':
                    task_desc_data_PP = {'record_type': 'task_desc',
                                         'desc': [lang_ann["task_desc"].strip()],
                                         'high_idx': [-4],
                                         'task_id': task_id,
                                         'action_sequence': ['PickupObject',
                                                             'PutObject']}
                    PP_related_objects = copy.deepcopy(get_task_related_objects(json_object))
                    # update position of agent as that of picked object
                    PP_agent_data = copy.deepcopy(agent_data)
                    PP_agent_data['position'] = copy.deepcopy(PP_related_objects[0]['position'])
                    # update position of receptable as position of object
                    PP_related_objects[1]['position'] = PP_related_objects[0]['position']
                    task_desc_data_PP['scene_description'] = [PP_agent_data] + PP_related_objects
                    task_descs_data.append(task_desc_data_PP)
                # -------------------------------------------------------- #

                count = 0
                # get the high-descs (sub-tasks) for the task
                high_descs = lang_ann["high_descs"]
                assignment_id = lang_ann["assignment_id"]

                # --------------------- HIGH-DESC ------------------------ #
                for high_desc in high_descs:
                    # certain actions signal that the agent's location has
                    # changed and requires it to be updated
                    update_agent_on_action(action_sequences, count, json_object)
                    high_desc_data = {
                        'record_type': 'high_desc',
                        'desc': [high_desc.strip()],
                        'high_idx': [count],
                        'assignment_id': assignment_id,
                        'parent_task_desc': lang_ann["task_desc"].strip(),
                        'action_sequence': [action_sequences[count]],
                        'scene_description': [copy.deepcopy(agent_data)] +
                                             get_relevant_high_desc_objects(
                                                 task_related_objects,
                                                 json_object, count)
                    }
                    # append to the list of high-descs that is maintained
                    high_descs_data.append(high_desc_data)
                    # maintain a separate list for creating multi-desc
                    task_high_desc.append(copy.deepcopy(high_desc_data))
                    count = count + 1
                # -------------------------------------------------------- #

                # create multi descs by passing all of the high-descs
                multi_descs_data = multi_descs_data + \
                                   get_multi_high_descs(task_high_desc)


            write_record(high_descs_data, multi_descs_data, task_descs_data,
                         is_training_record)


def write_record(high_descs_data, multi_descs_data, task_descs_data,
                 is_training_record):
    print_records(high_descs_data, multi_descs_data, task_descs_data)
    records = task_descs_data + high_descs_data + multi_descs_data

    if is_training_record:
        out_file = open(TRAINING_SET_FILE, APPEND)
    else:
        out_file = open(TESTING_SET_FILE, APPEND)

    for record in records:
        out_file.write(json.dumps(record) + '\n')

    out_file.close()


def print_records(high_descs_data, multi_descs_data, task_descs_data):
    # print out the data
    # print({'tasks': task_descs_data})
    # print({'high_descs': high_descs_data})
    # print({'multi_high_descs': multi_descs_data})

    # print out the counts
    # print('tasks       :', len(task_descs_data))
    # print('high_descs  :', len(high_descs_data))
    # print('multi_descs :', len(multi_descs_data))

    # maintain statistics
    global high_desc_count, multi_desc_count, task_desc_total
    high_desc_count = high_desc_count + len(high_descs_data)
    multi_desc_count = multi_desc_count + len(multi_descs_data)
    task_desc_total = task_desc_total + len(task_descs_data)


def update_agent_on_action(action_sequences, count, json_object):
    # The agents location is updated to the location of the
    # object that is picked when agent performs a pickup action
    if action_sequences[count] == PICKUP_ACTION:
        update_agent_data(json_object['plan']['high_pddl']
                          [count]['planner_action']['objectId'])
    # The agents location is updated the location of the
    # receptable when the agent performs a put-down action
    elif action_sequences[count] == PUTDOWN_ACTION:
        update_agent_data(json_object['plan']['high_pddl']
                          [count]['planner_action']
                          ['receptacleObjectId'])


def generate_training_data(folder_path):
    # get the list of all json files in the passed folder path
    files = get_json_file_paths(folder_path)
    for file_path in files:
        if PICK_AND_PLACE in file_path:
            parse_json_file(file_path)


# identify all the different types of floor plans within the training set for
# the task type PICK_AND_PLACE
def populate_floor_plans(folder_path):
    # get the list of all json files in the passed folder path
    files = get_json_file_paths(folder_path)
    global files_total
    for file_path in files:
        if PICK_AND_PLACE in file_path:
            with open(file_path) as json_file:
                json_object = json.load(json_file)
                if not json_object['pddl_params']['object_sliced']:
                    files_total = files_total + 1
                    floor_plan = get_floor_plan(json_object)
                    if floor_plan not in floor_plans:
                        floor_plans[floor_plan] = 1
                    else:
                        floor_plans[floor_plan] = floor_plans[floor_plan] + 1


def collect_statistics(folder_path):
    # get the list of all json files in the passed folder path
    files = get_json_file_paths(folder_path)
    for file_path in files:
        if PICK_AND_PLACE in file_path:
            with open(file_path) as json_file:
                json_object = json.load(json_file)
                language_annotations = json_object['turk_annotations']['anns']
                action_sequences = get_action_sequence(json_object)
                global a_seq_total, a_seq_count
                a_seq_total = a_seq_total + len(action_sequences)
                a_seq_count = a_seq_count + 1
                for lang_ann in language_annotations:
                    high_descs = lang_ann["high_descs"]
                    global high_desc_tot, high_desc_count
                    high_desc_tot = high_desc_tot + len(high_descs)
                    high_desc_count = high_desc_count + 1


populate_floor_plans(FOLDER_PATH)
print('Unique Floor Plans :', len(floor_plans))
print('Total Files        :', files_total)

# divide all floorplans between train and test sets
for plan in floor_plans:
    trial_count = trial_count + floor_plans[plan]
    if trial_count / files_total < TRAIN_PERCENT:
        train_floor_plans.append(plan)
    else:
        test_floor_plans.append(plan)

print('Train floor plans  :', len(train_floor_plans))
print('Test floor plans   :', len(test_floor_plans))

generate_training_data(FOLDER_PATH)

print('Tasks-descs   : ', task_desc_total)
print('High descs    : ', high_desc_count)
print('Multi descs   : ', multi_desc_count)
