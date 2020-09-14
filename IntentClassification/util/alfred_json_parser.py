
NO_OPERATION = 'NoOp'


# returns the action sequence in the passed ALFRED trajectory json file object
def get_action_sequence(json_object):
    high_pddl = json_object['plan']['high_pddl']
    action_sequence = []
    for action in high_pddl:
        operation = action['discrete_action']['action']
        if not operation == NO_OPERATION:
            action_sequence.append(operation)
    return action_sequence


# returns the list of all unique objects required for carrying out the task
# 'TASK' here refers to the goal of the trial
def get_task_related_objects(json_object):
    related_objects = []
    high_pddl = json_object['plan']['high_pddl']
    for action in high_pddl:
        related_object, receptable_object = get_object_and_receptacle(action)
        if not related_object is None and not related_object in related_objects:
            related_objects.append(related_object)
        if not receptable_object is None \
                and not receptable_object in related_objects:
            related_objects.append(receptable_object)
    return related_objects


# extracts the object and receptacle object if present from a
# single high_pddl_action
def get_object_and_receptacle(action):
    planner_action = action['planner_action']
    related_object = None
    receptacle_object = None
    if 'objectId' in planner_action:
        related_object_data = planner_action['objectId'].split('|')
        related_object = {
            'entityName': related_object_data[0],
            'object_type': 'simple',
            'relevant': 1,
            'position': [float(related_object_data[1]),
                         float(related_object_data[2]),
                         float(related_object_data[3])]
        }

    if 'receptacleObjectId' in planner_action:
        related_object_data = planner_action['receptacleObjectId'].split(
            '|')
        receptacle_object = {
            'entityName': related_object_data[0],
            'object_type': 'receptable',
            'relevant': 1,
            'position': [float(related_object_data[1]),
                         float(related_object_data[2]),
                         float(related_object_data[3])]
        }

    return related_object, receptacle_object


def get_floor_plan(json_object):
    return json_object['scene']['floor_plan']


# returns true if the trial is of the passed task type
def is_of_task_type(json_object, filter_on_task_type):
    return True
    if filter_on_task_type == '':
        return True
    elif 'task_type' in json_object and \
            json_object['task_type'] == filter_on_task_type:
        return True
    else:
        return False
