# IMPORT PACKAGES AND MODULES
# ///////////////////////////////////////////////////////////////
import json
import os
import logging
import numpy as np

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, fullpath: str):
        """
        Loads JSON data
        :rtype: object
        """
        super(DataLoader, self).__init__()
        self.fullpath = fullpath

        # DICTIONARY WITH SETTINGS
        # Just to have objects references
        self.items = {}
        self.position = []
        self.measurements = {}
        self.spatial_len = 0
        self.temporal_len = 0
        self.data = np.ndarray(shape=(0, 0))

        if not os.path.exists(self.fullpath):
            logger.warning(
                f"Incorrect path:\n{self.fullpath}"
            )
            self.items = None
        else:
            self.deserialize()
            self.get_data()

    # SERIALIZE JSON
    # ///////////////////////////////////////////////////////////////
    def serialize(self):
        # WRITE JSON FILE
        with open(self.fullpath, "w", encoding="utf-8") as write:
            json.dump(self.items, write, indent=4)

    # DESERIALIZE JSON
    # ///////////////////////////////////////////////////////////////
    def deserialize(self):
        # READ JSON FILE
        with open(self.fullpath, "r", encoding="utf-8") as reader:
            session = json.loads(reader.read())
            self.items = session

    # GET STRAIN DATA FROM JSON
    # ///////////////////////////////////////////////////////////////
    def get_data(self):
        assert {"measurements", "position"}.issubset(
            self.items.keys()
        ), "JSON file do not contains at least 'measurements' and 'position' keys"
        self.position = self.items["position"]
        self.measurements = self.items["measurements"]
        self.spatial_len = len(self.position)
        self.temporal_len = len(self.measurements.keys())
        self.data = np.zeros(shape=(self.temporal_len, self.spatial_len))

        for i in range(0, self.temporal_len):
            self.data[i, :] = self.measurements[str(i)]["strain"]
