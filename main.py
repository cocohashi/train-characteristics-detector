import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

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
    "-d", "--debug", action="store_true", help="Debug Mode"
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
day = 10

# Data Paths
root_path = os.path.abspath(os.path.join(os.getcwd(), "./.."))
data_path = os.path.join(root_path, "data", project_name)
data_path_ext = os.path.join(data_path, file_extension)
day_path = os.path.join(data_path_ext, str(year), f"{month:02}", f"{day:02}")
output_path = os.path.join(data_path_ext, "output")
base_path = os.path.join(data_path, "base")

# Tran classification flag
CLASSIFY_TRAINS = True

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

# Train Characteristic Schema
train_char_schema: dict[str, str | None] = {
    "datetime": None,
    "event": None,
    "status": "not-computed",
    "direction": None,
    "speed": None,
    "speed-magnitude": "kmh",
    "speed-error": None,
    "rail-id": None,
    "train-id": None,
    "confidence": None,
    "train-ids": train_ids,
    # "train-map": train_map
}

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
        "method": "gaussian"  # Allowed values only: "exponential", "gaussian", "reciprocal", "custom"
    },

    # Data Plotting
    "plot-matrix": {
        "section": 1,
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
        "positive-direction": "Cordoba - Malaga",
        "negative-direction": "Malaga - Cordoba"
    }
}


# -----------------------------------------------------------------------------------------------------------------
def make_data_dirs():
    # Exterior Data Path
    if not os.path.isdir(data_path_ext):
        os.makedirs(data_path_ext)

    # Input Day Path
    if not os.path.isdir(day_path):
        os.makedirs(day_path)

    # Output Day Path
    # output_day_path = os.path.join(output_path,  str(year), str(month), str(day))
    # logger.info(f"output_day_path: {output_day_path}")
    # if not os.path.isdir(output_day_path):
    #     os.makedirs(output_day_path)

    # Base path
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    return None


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

            base_train_ids = []
            for base_filename in base_filenames:
                filename_path = os.path.join(train_class_path, base_filename)
                base_train_id_data = np.load(filename_path)
                base_train_ids.append(base_train_id_data)

            base_data.append({"data": base_train_ids, "train-id": train_ids[i], "train-class": train_class})
    else:
        base_data = None

    return base_data


make_data_dirs()
base_data = get_and_check_base_data()


def get_train_characteristics(data: np.array, base_data: list = base_data, schema: dict = None) -> dict:
    """
    Computes train characteristics based on a given schema.

    :param data: Input numpy matrix containing Waterfall data
        :type data np.array()
    :param schema: A dictionary containing tool's output keys
        :type schema: object

    """
    if schema is None:
        schema = train_char_schema

    # Process the Waterfall to clean the noise
    signal_processor = SignalProcessor(data, **config)

    # Process Waterfall to obtain train characteristics
    char_detector = CharDetector(signal_processor, **config)

    # Update given schema with computed train characteristic values
    schema.update({
        "status": "computed",
        "event": "TRAIN",
        "direction": char_detector.direction,
        "rail-id": char_detector.rail_id,
        "speed": char_detector.speed,
        "speed-error": char_detector.speed_error})

    if base_data:
        char_detector.base_data = base_data
        train_id_info = char_detector.get_train_id_info()
        # logger.info(f"train_id_info: {train_id_info}")
        schema.update({
            "train-id": train_id_info['train-id'],
            "confidence": train_id_info['confidence'],
            # "train-class": train_id_info['train-class']
            "train-class": "Pending to be verified"
        })

    return {"train-char": schema, "signal-processor": signal_processor, "char-detector": char_detector}


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename

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
        section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.sobel_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_threshold_wf:
        section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.thr_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_mean_filter_wf:
        section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.mean_filter_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_mask_wf:
        section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.mask_sections[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_rail_view:
        section = config['plot-matrix']['section']
        data_plotter = DataPlotter(char_detector.rail_view[section], **config['plot-matrix'])
        data_plotter.plot_matrix()

    if args.show_train_track:
        config['plot-train-track']['title'] = f"{config['plot-train-track']['title']} - filename: {filename}"
        data_plotter = DataPlotter(char_detector.train_track, **config['plot-train-track'])
        data_plotter.plot_train_track()

    # Serialize data
    # data_loader.items.update(train_char)
    # data.fullpath = os.path.join(output_day_path, filename)
    # data.serialize()
