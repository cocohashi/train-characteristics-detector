import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.data_loader import DataLoader
from src.signal_processor import SignalProcessor
from src.data_plotter import DataPlotter
from src.char_detector import CharDetector

# Set Logger
plt.set_loglevel("warning")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set Argument Parser
parser = argparse.ArgumentParser(description="Tool to Compute Train Characteristics")
parser.add_argument(
    "-f", "--filename", type=str, help="Input Filename", required=True
)
parser.add_argument(
    "-dt", "--date", type=str, help="Input Date in '%Y-%m-%d' format", required=False
)
parser.add_argument(
    "-sec", "--section", type=int, help="Section that would be plotted", required=False
)

parser.add_argument(
    "-d", "--debug", action="store_true", help="Debug Mode"
)
parser.add_argument(
    "-s", "--serialize", action="store_true", help="Serialize JSON Data"
)
parser.add_argument(
    "-rw", "--show-raw-wf", action="store_true", help="Show Raw Waterfall"
)
parser.add_argument(
    "-fw", "--show-filtered-wf", action="store_true", help="Show Filtered Waterfall"
)
parser.add_argument(
    "-sw", "--show-sobel-wf", action="store_true", help="Show Sobel Waterfall"
)
parser.add_argument(
    "-tw", "--show-threshold-wf", action="store_true", help="Show Threshold Waterfall (one-hot-encoding)"
)
parser.add_argument(
    "-mfw", "--show-mean-filter-wf", action="store_true", help="Show Mean Filtered Waterfall (one-hot-encoding)"
)
parser.add_argument(
    "-mw", "--show-mask-wf", action="store_true", help="Show Computed Mask Waterfall (one-hot-encoding)"
)
parser.add_argument(
    "-rv", "--show-rail-view", action="store_true", help="Show Computed Rail View"
)
parser.add_argument(
    "-tt", "--show-train-track", action="store_true", help="Show Train Track"
)

# -----------------------------------------------------------------------------------------------------------------
# Confing Parameters
# -----------------------------------------------------------------------------------------------------------------
# Data directory names
project_name = "MC"
file_extension = "npy"  # Only "json" and "npy" allowed
year = 2023
month = 3
day = 9

# ----- Data Paths -----
root_path = os.path.abspath(os.path.join(os.getcwd(), "./.."))
data_path_dev = os.path.join(root_path, "data", project_name)
data_path_ext_dev = os.path.join(data_path_dev, file_extension)
day_path = os.path.join(data_path_ext_dev, str(year), f"{month:02}", f"{day:02}")
output_path = os.path.join(data_path_ext_dev, "output")
base_path = os.path.join(data_path_dev, "base")

# TODO: Dev base_path: "..data/{project_name}/{file_extension}"

# ----- Development Path -----
data_path = data_path_ext_dev
# ----------------------------

# ----- Production Path -----
# data_path = ""
# base_path = "" # optional
# ----------------------------

# Config Flags
CLASSIFY_TRAINS = True
TRAIN_CLASS_VERIFIED = False

# Train-map
train_map = {
    1: "S-102-6p",
    2: "S-102-8p",
    3: "S-103-u",
    4: "S-103-d",
    5: "S-104"
}
train_ids = list(train_map.keys())
train_classes = list(train_map.values())

# TODO:
#  In base path we MUST create all directories defined in 'train_map', and store there each train's base values.
#  Ex.
#  base_path tran-id 1: "../data/{project_name}/{file_extension}/S-102-6p"
#

# Train Characteristic Schema
train_char_schema = {
    "datetime": None,
    "event": None,
    "status": "not-computed",
    "direction": None,
    "speed": None,
    "speed-magnitude": "kmh",
    "speed-error": None,
    "rail-id": None,
    "rail-id-confidence": None,
    "train-id": None,
    "train-id-confidence": None,
    "train-ids": train_ids,
    # "train-map": train_map
}

# Output key were serialize data. This key may exist or not in input data.
output_key = "info"

# Signal processing
config = {
    # Signal Processor
    "signal": {
        "N": 5,  # int: Downsampling-Factor: Number of Samples to be downsampled
        "f_order": 4,  # int: The order of the Butterworth filter.
        "Wn-mask": [10, 99],  # int or list: Cutoff frequencies of Butterworth filter
        "Wn-class": [0.8, 5],  # int or list: Cutoff frequencies of Butterworth filter
        "btype": "bp",  # str: Butterworth filter type. {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        "fs": 1000,  # int: The sampling frequency of the digital system in Hz.
    },
    # Train Characteristics Detector
    "char-detector": {
        "section-limit": 108,
        "section-num": 2,
        "mask-width": 12000,
        "thr-perc": 0.1,
        "mf-window": 20,
        "no-train-event-thr": 0.02,
        "double-train-event-thr": 0.2,
        "double-train-event-max-dist": 1000,
        "upper-speed-limit": 400,
        "lower-speed-limit": 150,
        "decimal": 3,
        "train-confidence-perc-limit": 0.2,  # A classified train, with a confidence percentage value lower than
        # this limit, will be considered as "unknown" train-class and train-id.
        "method": "gaussian"  # Allowed values only: "exponential", "gaussian", "reciprocal", "custom"
    },

    # Events
    "event": {
        "train": "train",
        "no-train": "no-train",
        "double-train": "double-train",
        "slow-train": "slow-train",
        "unknown": "unknown"
    },

    # Data Plotting
    "plot-matrix": {
        "section": 0,
        "vmin": None,
        "vmax": None,
        "xlabel": "x-axis (samples)",
        "ylabel": "y-axis (samples)",
        "title": "",
        "cmap": "seismic",
        "figsize": None,
        "extent": None
    },

    "plot-train-track": {
        "figsize": None,
        "xlabel": "x-axis (samples)",
        "ylabel": "y-axis (samples)",
        "title": "Train Track",
    },

    "schema": {
        "positive-direction": "Cordoba -> Malaga",
        "negative-direction": "Malaga -> Cordoba"
    }
}


# -----------------------------------------------------------------------------------------------------------------
def make_data_dirs(data_path=data_path):
    # Exterior Data Path
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # Input Day Path
    if not os.path.isdir(day_path):
        os.makedirs(day_path)

    # Output Day Path
    output_day_path = os.path.join(output_path, str(year), str(month), str(day))
    logger.info(f"output_day_path: {output_day_path}")
    if not os.path.isdir(output_day_path):
        os.makedirs(output_day_path)

    # Base path
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    return {"data_path": data_path,
            "day_path": day_path,
            "output_day_path": output_day_path,
            "base_path": base_path}


def get_and_check_base_data():
    global CLASSIFY_TRAINS
    base_path_train_class_pool = [dr for dr in os.listdir(base_path)]

    # Not found train-track directories
    if len(base_path_train_class_pool) < 1:
        logger.warning(f"Base path is empty. Train-id cannot be computed.")
        logger.warning(f"Base path: {base_path}")
        CLASSIFY_TRAINS = False

    else:
        for not_found_dir in list(set(train_classes) - set(base_path_train_class_pool)):
            logger.warning(f" '{not_found_dir}' train-id defined and not found")
            CLASSIFY_TRAINS = False

        for ignore_dir in list(set(base_path_train_class_pool) - set(train_classes)):
            logger.warning(f" '{ignore_dir}' is defined in base-path but not in specs. It will be ignored.")
            CLASSIFY_TRAINS = False

    logger.info(f"CLASSIFY_TRAINS: {CLASSIFY_TRAINS}")

    # Get Base Train-Id's data
    if CLASSIFY_TRAINS:
        base_data = []
        for i, train_class in enumerate(train_classes):
            train_class_path = os.path.join(base_path, train_class)
            base_filenames = [file for file in os.listdir(train_class_path) if
                              file.split('.')[-1] == "npy"]
            # --- Debug ---
            # logger.info(f"Files for {train_id}\n{base_filenames}")
            # -------------

            base_train_class_data_list = []
            for base_filename in base_filenames:
                filename_path = os.path.join(train_class_path, base_filename)
                base_train_class_data = np.load(filename_path)
                base_train_class_data_list.append(base_train_class_data)

            base_data.append({"data": base_train_class_data_list, "train-id": train_ids[i], "train-class": train_class})
    else:
        base_data = None

    return base_data


dir_paths = make_data_dirs()
base_data_values = get_and_check_base_data()


def get_train_characteristics(data: np.array, base_data: list = None, schema: dict = None) -> dict:
    """
    Computes train characteristics based on a given schema.

    :param data: Input numpy matrix containing Waterfall data
        :type data np.array()
    :param schema: A dictionary containing tool's output keys
        :type schema: object

    """
    if base_data is None:
        base_data = base_data_values

    if schema is None:
        schema = train_char_schema

    # Process the Waterfall to clean the noise
    signal_processor = SignalProcessor(data, **config)

    # Process Waterfall to obtain train characteristics
    char_detector = CharDetector(signal_processor, **config)

    # Update given schema with computed train characteristic values
    schema.update({
        "status": "computed",
        "event": char_detector.event,
        "direction": char_detector.direction,
        "rail-id": char_detector.rail_id,
        "rail-id-confidence": char_detector.rail_id_confidence,
        "speed": char_detector.speed,
        "speed-error": char_detector.speed_error,
        "train-id": None,
        "train-id-confidence": None
    })

    if base_data:
        char_detector.base_data = base_data
        if char_detector.event == config['event']['train']:
            train_id_info = char_detector.get_train_id_info()
            if train_id_info['confidence'] < config['char-detector']['train-confidence-perc-limit']:
                train_id_info.update({"train-id": "unknown", "train-class": "unknown"})
            schema.update({
                "train-id": train_id_info['train-id'],
                "train-id-confidence": train_id_info['confidence'],
            })
            if TRAIN_CLASS_VERIFIED:
                schema.update({"train-class": train_id_info['train-class']})
            else:
                schema.update({"train-class": "Pending to be verified"})

    # Reset char_detector object values

    return {"train-char": schema, "signal-processor": signal_processor, "char-detector": char_detector}


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename
    section = None

    # Check date
    if args.date:
        date_format = "%Y-%m-%d"
        assert (
            bool(datetime.strptime(args.date, date_format))
        ), f"date does not follows {date_format} date format."
        year, month, day = [args.date.split('-')[x] for x in range(3)]
        day_path = os.path.join(data_path, str(year), f"{int(month):02}", f"{int(day):02}")

    if args.section:
        section = args.section
        section_num = config['char-detector']['section-num']
        assert (
            bool(section < section_num)
        ), f"The given section should be less than {section_num}"

    # Check data entry
    assert (
            filename.split(".")[-1] == file_extension
    ), (f"Defined '{file_extension}' file-extension in config, do not matches with given file-extension"
        f" '{filename.split('.')[-1]}'.")

    assert (
        os.path.exists(day_path)
    ), f"data-path: '{day_path}' not found"

    available_filenames = [filename for filename in os.listdir(day_path)]

    assert (
            filename in set(available_filenames)
    ), f"filename '{filename}' does not exist in path: {day_path}"

    # Define development filepath
    file_path = os.path.join(day_path, filename)

    # Load data
    data_loader = DataLoader(file_path)

    logger.info(data_loader.data.shape)
    # Get train characteristics
    output = get_train_characteristics(data_loader.data)

    # Destructuring
    train_char = output['train-char']
    signal_processor = output['signal-processor']
    char_detector = output['char-detector']

    # Debug
    if args.debug:
        logger.debug(f"train-characteristic-values:\n{train_char}")

    if args.show_raw_wf:
        data_plotter = DataPlotter(data_loader.data, **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_filtered_wf:
        data_plotter = DataPlotter(signal_processor.filtered_data, **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_sobel_wf:
        if not section:
            section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.sobel_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_threshold_wf:
        if not section:
            section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.thr_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_mean_filter_wf:
        if not section:
            section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.mean_filter_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_mask_wf:
        if not section:
            section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.mask_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_rail_view:
        if not section:
            section = config['plot-matrix']['section']
        if char_detector.rail_view:
            data_plotter = DataPlotter(char_detector.rail_view[section], **config['plot-matrix'])
            data_plotter.plot_matrix()

    if args.show_train_track:
        config['plot-train-track']['title'] = f"{config['plot-train-track']['title']} - filename: {filename}"
        data_plotter = DataPlotter(char_detector.train_track, **config['plot-train-track'])
        data_plotter.plot_train_track()

    if args.serialize:
        logger.info("Serializing data...")
        output_day_path = dir_paths['output_day_path']
        if data_loader.items.get(output_key):
            data_loader.items[output_key].update(train_char)
            data_loader.fullpath = os.path.join(output_day_path, filename)
            data_loader.serialize()
        else:
            logger.warning(f"output_key: {output_key} wasn't created. Serializing data to JSON anyway.")
            data_loader.items.update({output_key: train_char})
            data_loader.fullpath = os.path.join(output_day_path, filename)
            data_loader.serialize()
        logger.info(
            f"Serialized.\n"
            f"Train characteristics updated in: '{filename}' JSON file, saved in path:"
            f"\n{output_day_path}")
