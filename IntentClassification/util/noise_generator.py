# Author     :  Karun Mathew
# Student Id :  1007247
#
# This program helps to generate data samples with a small amount of uniform noise
# The noise is added to visual features alone

import random
import math
from util.apputil import MAX_ANGLE, ARM_LENGTH, MAX_DISTANCE, PRECISION


def get_unknown_or_noise():
    if random.uniform(0, 100) > 50:
        return -1
    else:
        return round(random.uniform(0, MAX_DISTANCE), PRECISION)


# generates visual features
# 1. Distance from agent to object
# 2. Distance from agent to receptacle
# 3. Distance between agent and receptacle
# 4. Angle between object and the direction the agent is facing
def get_pick_up_noise():
    return str(round(random.uniform(0.6, MAX_DISTANCE), PRECISION)) + ' ' + str(get_unknown_or_noise()) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(-1, 1), PRECISION))


# this effectively generates an adversarial sample
# adversarial samples are found close to the decision boundary and so a greater number
# of adversarial samples are generated in order to help the agent understand the decision
# boundary
def get_close_pick_up_noise():
    return str(round(random.uniform(ARM_LENGTH, 1), PRECISION)) + ' ' + str(get_unknown_or_noise()) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(-1, 1), PRECISION))


def get_close_put_down_noise():
    return str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(ARM_LENGTH, 1), PRECISION)) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(-1, 1), PRECISION))


def get_put_down_noise():
    return str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(ARM_LENGTH, MAX_DISTANCE), PRECISION)) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(-1, 1), PRECISION))


def add_noise(action_sequence_string, visual_data):
    if action_sequence_string.strip() == 'PickupObject':
        visual_data[0] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        # noise
        visual_data[1] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    if action_sequence_string.strip() == 'PutObject':
        visual_data[1] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        # noise
        visual_data[0] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    if action_sequence_string.strip() == 'GotoLocation PickupObject PutObject':
        visual_data[2] = round(random.uniform(0, ARM_LENGTH), PRECISION)

    if action_sequence_string.strip() == 'GotoLocation PutObject':
        visual_data[1] = round(visual_data[1] + round(random.uniform(-0.2, 0.2), PRECISION), PRECISION)
        # noise
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(-1, 1), PRECISION)

    if action_sequence_string.strip() == 'PickupObject PutObject':
        visual_data[0] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        visual_data[1] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        visual_data[2] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    if action_sequence_string.strip() == 'PickupObject GotoLocation PutObject':
        visual_data[0] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), PRECISION), PRECISION)
        visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    if action_sequence_string.strip() == 'PickupObject GotoLocation':
        visual_data[0] = round(random.uniform(0, ARM_LENGTH), PRECISION)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), PRECISION), PRECISION)
        visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    # if the first action is PickupObject, consider the orientation info as well
    # generate a sample for the scenario, when the agent needs to turn to perform
    # the pick or put action
    if action_sequence_string.strip().startswith('PickupObject') or \
            action_sequence_string.strip().startswith('PutObject'):

        if random.random() < 0.5:
            # Needs to rotate to pick the object
            visual_data[3] = round(random.uniform(-1, math.cos(math.radians(MAX_ANGLE))), PRECISION)
            action_sequence_string = 'RotateAgent ' + action_sequence_string.strip()
        else:
            visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), PRECISION)

    action_sequence_string = action_sequence_string.strip()
    return action_sequence_string, visual_data
