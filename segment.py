import copy
from typing import Any


NUM_OBSERVATIONS = 5
NUM_ACTIONS = 1
POSITION_DIMENSIONS = 2

class StateActionPair:
    def __init__(self, radars, action, position, alive):
        if len(position) != POSITION_DIMENSIONS:
            raise ValueError("position must be 2 floats")
        try:
            self.radars = radars.tolist()
        except AttributeError:
            self.radars = radars
        try:
            self.action = action.tolist()
        except AttributeError:
            self.action = action
        try:
            self.position = position.tolist()
        except AttributeError:
            self.position = position

        self.alive = alive

    def __iter__(self):
        return iter(self.radars + [self.action] + self.position)

    def __getitem__(self, index):
        if 0 <= index < NUM_OBSERVATIONS:
            return self.radars[index]
        elif index == NUM_OBSERVATIONS:
            return self.action
        elif NUM_OBSERVATIONS + NUM_ACTIONS <= index < NUM_OBSERVATIONS + NUM_ACTIONS + POSITION_DIMENSIONS:
            return self.position[index - NUM_OBSERVATIONS - 1]
        else:
            raise IndexError("Index out of range")

    def __repr__(self):
        return (
            "StateActionPair: <Radars: "
            + str([radar for radar in self.radars])
            + ", Action: "
            + str(self.action)
            + ", Position: "
            + str(self.position)
            + ">"
        )

    def __len__(self):
        return NUM_OBSERVATIONS + NUM_ACTIONS + POSITION_DIMENSIONS
    
    def __deepcopy__(self, memo: dict[str, Any]):
        return StateActionPair(
            copy.deepcopy(self.radars, memo),
            copy.deepcopy(self.action, memo),
            copy.deepcopy(self.position, memo),
            self.alive,
        )
