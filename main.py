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
# Metadata
file_extension = "npy"  # Only "json" and "npy" allowed
year = 2023
month = 3
day = 8

# Paths
root_path = os.path.abspath(os.path.join(os.getcwd(), "./.."))
data_path = os.path.join(root_path, "data", "MC", file_extension)
day_path = os.path.join(data_path, str(year), f"{month:02}", f"{day:02}")
output_path = os.path.join(data_path, "output")

# Train Characteristic Schema
train_char_schema: dict[str, str | None] = {
    "datetime": None,
    "event": None,
    "status": "not-computed",
    "direction": None,
    "rail-id": None,
    "train-id": None,
    "speed": None,
    "speed-magnitude": "kmh",
    "speed-error": None
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
        "mf-window": 20
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

def get_train_characteristics(data: np.array, schema: dict = None) -> dict:
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
    schema.update({"direction": char_detector.direction,
                   "rail-id": char_detector.rail_id,
                   "speed": char_detector.speed,
                   "speed-error": char_detector.speed_error})

    return {"train_char": schema, "signal_processor": signal_processor, "char_detector": char_detector}


def make_output_data_dirs():
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # Output paths
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_year_path = os.path.join(output_path, str(year))
    if not os.path.isdir(output_year_path):
        os.mkdir(output_year_path)

    output_month_path = os.path.join(output_year_path, str(month))
    if not os.path.isdir(output_month_path):
        os.mkdir(output_month_path)

    output_day_path = os.path.join(output_month_path, str(day))
    if not os.path.isdir(output_day_path):
        os.mkdir(output_day_path)

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename
    make_output_data_dirs()

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
    train_char = output['train_char']
    signal_processor = output['signal_processor']
    char_detector = output['char_detector']

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
