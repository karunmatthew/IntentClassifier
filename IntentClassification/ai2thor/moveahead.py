# These are test classes that were to used to check if the ai2thor was
# properly installed
# This program tries to make the agent take a series of simple actions

from ai2thor.controller import Controller
import time

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430

controller = Controller(scene='FloorPlan28', gridSize=0.25)
print(controller.initialization_parameters)
time.sleep(10)
print('Start')
time.sleep(5)
event = controller.step(action='MoveAhead')
print(event.metadata)
print('Moved Ahead')
for i in range(5):
    event = controller.step(action='MoveRight', moveMagnitude=1 * 1)
    time.sleep(5)
    print(event.metadata)
    event = controller.step(action='MoveAhead', moveMagnitude=0.25 * 1)
    # current metadata dictionary that includes the state of the scene


# Numpy Array - shape (width, height, channels), channels are in RGB order
event.frame

# Numpy Array in BGR order suitable for use with OpenCV
event.cv2image

# current metadata dictionary that includes the state of the scene
print(event.metadata)

# shuts down the controller
controller.stop()

