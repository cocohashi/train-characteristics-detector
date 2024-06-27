import os
import logging

import json
import numpy as np

# -------------------------------------------------------------------------------------------------------------------
# Set Logger
# -------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler() if os.environ['ENVIRONMENT'] == 'develop' else logging.FileHandler('main.log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
# -----------------------------------------------------------------------------------------------------------------


class DataLoader:
    def __init__(self, fullpath: str):
        """
        Loads JSON or Numpy data
        :rtype: object
        """
        super(DataLoader, self).__init__()
        self.fullpath = fullpath
        self.filename = os.path.basename(fullpath)
        self.extension = self.filename.split('.')[1]
        # DICTIONARY WITH SETTINGS
        # Just to have objects references
        self.items = {}
        self.position = []
        self.measurements = {}
        self.spatial_len = 0
        self.temporal_len = 0
        self.data = np.ndarray(shape=(0, 0))
        self.base_data = None
        self.rail_view_data = None

        if not os.path.exists(self.fullpath):
            logger.warning(
                f"Incorrect path:\n{self.fullpath}"
            )
            self.items = None
        else:
            if len(self.filename.split(".json")) > 1:
                self.deserialize()
                self.get_json_data()

            elif len(self.filename.split(".npy")) > 1:
                self.get_npy_data()

            else:
                logger.warning(
                    f"Incorrect file extension given: '{self.extension}'. Only 'json' or 'npy' is allowed."
                )

    # SERIALIZE JSON
    # ///////////////////////////////////////////////////////////////
    def convert_dict_values_to_str(self):
        self.items = {k: str(v) for k, v in self.items.items()}

    def serialize(self):
        self.convert_dict_values_to_str()
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
    def get_json_data(self):
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

    # GET STRAIN DATA FROM NUMPY ARRAY
    # ///////////////////////////////////////////////////////////////
    def get_npy_data(self):
        self.data = np.load(self.fullpath)
        self.temporal_len = self.data.shape[0]
        self.spatial_len = self.data.shape[1]

    # DESERIALIZE JSON
    # ///////////////////////////////////////////////////////////////
    def deserialize_npy(self, base_data=False, rail_view_data=False):
        if base_data:
            np.save(self.fullpath, self.base_data)
        elif rail_view_data:
            np.save(self.fullpath, self.rail_view_data)
        else:
            np.save(self.fullpath, self.data)
