from ai2thor.controller import Controller
controller = Controller(scene='FloorPlan28', gridSize=0.25)

# change starting locations
controller.step(action='Teleport', x=-2.5, y=0.900998235, z=-3.0)
controller.step(action='LookDown')
event = controller.step(action='Rotate', rotation=180)

# in FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        # pick up the mug
        event = controller.step(action='PickupObject',
                                objectId=o['objectId'],
                                raise_for_failure=True)
        mug_object_id = o['objectId']
        break

# the agent now has the Mug in its inventory
# to put it into the Microwave, we need to open the microwave first

# move to the microwave
event = controller.step(action='LookUp')
event = controller.step(action='RotateLeft')
event = controller.step(action='MoveLeft', moveMagnitude=0.25 * 4)
event = controller.step(action='MoveAhead', moveMagnitude=0.25 * 6)

# the agent should now be looking at the microwave
for o in event.metadata['objects']:
    if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
        # open the microwave
        event = controller.step(action='OpenObject',
                                objectId=o['objectId'],
                                raise_for_failure=True)
        receptacle_object_id = o['objectId']
        break

# put the object in the microwave
event = controller.step(
    action='PutObject',
    receptacleObjectId=receptacle_object_id,
    objectId=mug_object_id,
    raise_for_failure=True)

# close the microwave
event = controller.step(
    action='CloseObject',
    objectId=receptacle_object_id,
    raise_for_failure=True)

