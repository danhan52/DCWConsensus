import numpy as np


class LineMatcher():

    def __init__(self, _yTolerance, _xTolerance=None):
        self.currentCoords = (-1, -1, -1, -1)
        self.yTolerance = _yTolerance
        self.xTolerance = _xTolerance

    def compare(self, lineCoords):
        # check if y coords are further than threshold
        different = np.abs(
            lineCoords[1] - self.currentCoords[1]) > self.yTolerance or np.abs(
                lineCoords[3] - self.currentCoords[3]) > self.yTolerance
        # if y are close and we care about x, look at x distance
        if not different and self.xTolerance is not None:
            different = different and (
                np.abs(lineCoords[0] - self.currentCoords[0]) > self.xTolerance
                or np.abs(lineCoords[2] - self.currentCoords[2]) >
                self.xTolerance)
        # change self location if different
        if different:
            self.setCurrentCoords(lineCoords)
            return True
        else:
            return False

    def setCurrentCoords(self, _newCoords):
        self.currentCoords = _newCoords
