## lookup:kitchen_items
data/kitchen_items.txt

## lookup:stationary_items
data/stationary_items.txt

## intent:GRASP
- pick
- take
- obtain
- Pick up the plate that is to the left of the cabbage.
- Pick up the patterned plate.
- Pick up the yellow handled knife that's in front of you.
- Pick up one of the slices of bread previously cut, on the [counter](stationary_items).
- Pick up the egg from the back of the [counter](stationary_items).

## intent:RELEASE
- place
- keep
- put
- drop
- put the [fork](kitchen_items) down
- Place the [plate](kitchen_items) on the black table, next to the [knife](kitchen_items).
- Put the knife in the sink near the [apple](kitchen_items).
- Put the knife in the [sink](stationary_items).
- Put the apple in the [microwave](stationary_items).
- Put the plate in the [cabinet](stationary_items) above the sink.


## intent:MOVE_RIGHT
- go right
- move right
- turn right
- walk right
- navigate right
- Turn right and walk to the [microwave](stationary_items).
- Turn right and take a step to your right.


## intent:MOVE_LEFT
- go left
- move left
- turn left
- walk left
- navigate left
- Move left, to face the left half of the stove.
- Turn left and go to the [counter](stationary_items).


## intent:MOVE_FORWARD
- go forwards
- move forwards
- go straight
- go
- walk to the [fridge](stationary_items)
- Go to the [fridge](stationary_items)
- Walk forward to the corner counter area.
- Go forward and get in front of the loaf of bread that is next to the salt


## intent:MOVE_BACK
- go backward
- move backwards
- turn around
- go behind
- Go to the white table behind you.
- Turn around and move to the [counter](stationary_items) on the far left of the wall
- Turn around and go to the red garbage can, in the corner, across from you.
- Turn around and walk to the [cabinet](stationary_items) under the counter.


## intent:TRANSPORT
- Take the cup to the counter
- Carry the spoon to the table


## intent:NOT_SUPPORTED
- Heat the apple in the microwave
- Open the door of the oven
- Look down at the rubbish bin