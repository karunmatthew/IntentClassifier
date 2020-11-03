import random
import math
from util.apputil import MAX_ANGLE


def get_unknown_or_noise():
    if random.uniform(0, 100) > 50:
        return -1
    else:
        return round(random.uniform(0, 8), 2)


def get_pick_up_noise():
    return str(round(random.uniform(0.6, 8), 2)) + ' ' + str(get_unknown_or_noise()) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0, 1), 2))


def get_put_down_noise():
    return str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0.6, 8), 2)) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0, 1), 2))


def add_noise(action_sequence_string, visual_data):
    if action_sequence_string.strip() == 'PickupObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        # noise
        visual_data[1] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(0, 1), 2)

    if action_sequence_string.strip() == 'PutObject':
        visual_data[1] = round(random.uniform(0, 0.5), 2)
        # noise
        visual_data[0] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(0, 1), 2)

    if action_sequence_string.strip() == 'GotoLocation PickupObject PutObject':
        visual_data[2] = round(random.uniform(0, 0.5), 2)

    if action_sequence_string.strip() == 'GotoLocation PutObject':
        visual_data[1] = round(visual_data[1] + round(random.uniform(-0.2, 0.2), 2), 2)
        # noise
        visual_data[2] = get_unknown_or_noise()

    if action_sequence_string.strip() == 'PickupObject PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[1] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(random.uniform(0, 0.5), 2)

    if action_sequence_string.strip() == 'PickupObject GotoLocation PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)

    if action_sequence_string.strip() == 'PickupObject GotoLocation':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)

    # if the first action is PickupObject, consider the orientation info as well
    if action_sequence_string.strip().startswith('PickupObject') or \
            action_sequence_string.strip().startswith('PutObject'):

        if random.random() < 0.5:
            # visual_data[3] = round(random.uniform(-1, 0), 2)
            visual_data[3] = round(random.uniform(-1, math.cos(math.radians(MAX_ANGLE))), 2)
        else:
            # visual_data[3] = round(random.uniform(0, 1), 2)
            visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), 2)

        if math.degrees(math.acos(visual_data[3])) > MAX_ANGLE:
            action_sequence_string = 'RotateAgent ' + action_sequence_string.strip()
            action_sequence_string = action_sequence_string.strip()

    action_sequence_string = action_sequence_string.strip()
    return action_sequence_string, visual_data
