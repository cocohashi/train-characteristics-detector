import os
import logging
import numpy as np

from typing import Callable
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from scipy import signal
from dtaidistance import dtw
from .signal_processor import SignalProcessor

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


class CharDetector(SignalProcessor):
    def __init__(self, signal_processor, **config):
        super(SignalProcessor, self).__init__()

        self.signal_processor = signal_processor
        self.filtered_data = signal_processor.filtered_data  # Cut-off freq: Wn-mask
        self.N = signal_processor.N
        self.dt = signal_processor.dt

        self.signal_processor.Wn = config['signal']['Wn-class']  # Change Cut-off freq: Wn-class
        self.section_limit = config['char-detector']['section-limit']
        self.section_num = config['char-detector']['section-num']
        self.sub_section_num = config['char-detector']['sub-section-num']
        self.mask_width = config['char-detector']['mask-width'] / self.N
        self.thr_perc = config['char-detector']['thr-perc']
        self.mf_window = config['char-detector']['mf-window']
        self.no_train_event_thr = config['char-detector']['no-train-event-thr']
        self.double_train_event_thr = config['char-detector']['double-train-event-thr']
        self.double_train_event_max_dist = config['char-detector']['double-train-event-max-dist']
        self.upper_speed_limit = config['char-detector']['upper-speed-limit']
        self.lower_speed_limit = config['char-detector']['lower-speed-limit']
        self.decimal = config['char-detector']['decimal']
        self.multi_regression = config['char-detector']['multi-regression']
        self.multi_regression_epochs = config['char-detector']['multi-regression-epochs']
        self.multi_regression_mask_width_margin = config['char-detector']['multi-regression-mask-width-margin']

        self.method = config['char-detector']['method']

        self.positive_direction = config['schema']['positive-direction']
        self.negative_direction = config['schema']['negative-direction']
        self.event = config['event']['train']
        self.no_train_event = config['event']['no-train']
        self.double_train_event = config['event']['double-train']
        self.unknown_event = config['event']['unknown']
        self.slow_train = config['event']['slow-train']

        self.direction = ""
        self.rail_id = None
        self.energy_diff = None
        self.rail_id_confidence = None
        self.speed = 0
        self.speed_error = 0
        self.train_track = []
        self.base_data = None
        self.rail_view = None
        self.train_id_info = None

        self.sobel_sections = None
        self.mean_filter_sections = None
        self.thr_sections = None
        self.coordinates_sections = None
        self.linear_reg_params = None
        self.previous_linear_reg_params = None
        self.mask_sections = None
        self.masked_sections = None
        self.mask_sub_sections = []

        # Slice waterfall in different spatial sections
        self.sections = self.get_sections()

        # Get mean filter of each section
        self.get_mean_filter_sections(input_data=self.sections)

        # Check 'NO-TRAIN' event
        if self.check_event(event=self.no_train_event):
            self.event = self.no_train_event

        # Check 'DOUBLE-TRAIN' event
        elif self.check_event(event=self.double_train_event):
            self.event = self.double_train_event

        # Compute characteristics for 'TRAIN' event
        else:
            # Get mask of each section
            self.get_mask_sections()

            # Get mask subsections
            if self.sub_section_num:
                self.get_mask_subsections()

            # Filter data again using the filter defined in Wn-class
            self.filtered_data = self.signal_processor.butterworth_filter()  # Apply Cut-off freq: Wn-class
            self.sections = self.get_sections()

            # Get rail view each section
            self.get_rail_view(apply_mask=self.sections)

            # Get train characteristics
            self.get_direction()
            self.get_rail_id()
            self.get_rail_id_confidence()
            self.get_speed(decimal=self.decimal)
            self.get_train_track()

            if self.multi_regression:
                for x in range(self.multi_regression_epochs):
                    self.mask_width = self.mask_width * (1 - self.multi_regression_mask_width_margin)
                    self.previous_linear_reg_params = self.linear_reg_params
                    logger.info(f"previous_linear_reg_params: {self.previous_linear_reg_params}")
                    self.get_mean_filter_sections(input_data=self.rail_view)
                    self.get_mask_sections()
                    self.get_rail_view(apply_mask=self.rail_view)
                    logger.info(f"linear_reg_params: {self.linear_reg_params}")
                    self.update_slopes()
                    self.get_direction()
                    self.get_rail_id()
                    self.get_rail_id_confidence()
                    self.get_speed(decimal=self.decimal)
                    self.get_train_track()

            # Depending on computed speed we could face an "unknown" or "slow-train" events
            if self.speed > self.upper_speed_limit:
                self.event = self.unknown_event
            elif self.speed < self.lower_speed_limit:
                self.event = self.slow_train

    def get_mean_filter_sections(self, input_data):
        # Compute sobel and threshold filters to each section
        self.sobel_sections = [self.sobel_filter(data) for data in input_data]
        self.thr_sections = [self.thresholding(data) for data in self.sobel_sections]

        # Get mask of each section
        self.mean_filter_sections = [self.mean_filter(data) for data in self.thr_sections]

    def get_mask_sections(self):
        self.coordinates_sections = [self.one_hot_to_coordinates(data) for data in self.mean_filter_sections]
        self.linear_reg_params = [self.linear_regression(coordinate) for coordinate in self.coordinates_sections]
        self.mask_sections = [self.get_mask(self.thr_sections[x], **self.linear_reg_params[x]) for x in
                              range(self.section_num)]
        logger.info(f"get mask_sections.shape: {self.mask_sections[0].shape}")

    def get_rail_view(self, apply_mask) -> None:
        if apply_mask is None:
            apply_mask = None

        logger.info(f"apply_mask.shape: {apply_mask[0].shape}")
        logger.info(f"get mask_sections.shape: {self.mask_sections[0].shape}")
        # Apply mask to each section
        self.masked_sections = [np.multiply(apply_mask[x], self.mask_sections[x]) for x in
                                range(self.section_num)]
        self.masked_sections = [self.auto_crop(data) for data in self.masked_sections]
        self.rail_view = [self.filter_zeros(data) for data in self.masked_sections]

    def update_slopes(self):
        slopes = [section.get('slope') for section in self.linear_reg_params]
        previous_slopes = [section.get('slope') for section in self.previous_linear_reg_params]
        logger.info(f"previous_slopes: {previous_slopes}")

        if slopes and previous_slopes:
            new_slopes = [sum(x) for x in zip(slopes, previous_slopes)]
            logger.info(f"slopes: {slopes}")
            logger.info(f"new_slopes: {new_slopes}")
            for i, d in enumerate(self.linear_reg_params):
                d.update((k, new_slopes[i]) for k, v in d.items() if k == "slope")

            # logger.info(f"updated: {self.linear_reg_params}")

    def get_sections(self):
        """
        Subdivides matrix into "n" Spatial Sections by a given "section limit"
        :return: 2D Numpy array
        """
        s_lim = self.section_limit
        n = self.section_num
        return [self.filtered_data[:, x * s_lim: (x + 1) * s_lim] for x in range(n)]

    @staticmethod
    def sobel_filter(data, sigma=1):
        """
        Applies a sobel filter that highlights vertical and horizontal borders for a given 2D numpy array.
        :param data Waterfall matrix Spatial Section
        :param sigma: Sobel's filter parameter that determines output amplitude range
        :return: 2D Numpy array
        """
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(data, Kx)
        Iy = ndimage.filters.convolve(data, Ky)

        G = np.hypot(Ix, Iy)
        sobel = G / G.max() * sigma

        return sobel

    def thresholding(self, data):
        """
        Applies thresholding to obtain one-hot-encoding 2D matrix.
        :param data: Sobel matrix Spatial Section
        :return: 2D Numpy array
        """
        min = data.min()
        max = data.max()
        thr = min + (max - min) * self.thr_perc

        output = np.zeros(data.shape)
        mask_x, mask_y = np.where(data >= thr)
        output[mask_x, mask_y] = 1

        return output

    def mean_filter(self, data):
        """
        Makes the mean filter of the columns first and then the lines to reduce noise and obtain much more cleaner
        train trace.
        The best results are obtained computing this procedure to a one-hot-encoding matrix.
        :param data: 2D numpy array (in one-hot-encoding format)
        :param mf_window: mean filter windows length
        :return: 2D numpy array (in one-hot-encoding format)
        """
        column_filter = ndimage.median_filter(data, size=(1, self.mf_window))
        return ndimage.median_filter(column_filter, size=(self.mf_window, 1))

    def check_event(self, event):
        if event == self.no_train_event:
            mean = np.mean([np.mean(self.mean_filter_sections[x]) for x in range(self.section_num)])
            return bool(mean < self.no_train_event_thr)
        if event == self.double_train_event:
            results = []
            for section in range(self.section_num):
                for pos in range(self.mean_filter_sections[section].shape[1]):
                    col_values = self.mean_filter_sections[section][:, pos]
                    t_diff = np.diff(col_values.nonzero()[0])
                    t_dist = np.array([1 if x > self.double_train_event_max_dist else 0 for x in t_diff])
                    results.append(np.sum(t_dist))
            return bool(np.mean(results) > self.double_train_event_thr)

    @staticmethod
    def one_hot_to_coordinates(data):
        """
        Obtains x and y coordinate axis from one-hot-encoded 2D array matrix
        :param data: 2D numpy array (in one-hot-encoding format)
        :return: 2D numpy array of shape (N,2) where N depends on the number of "1's" in the matrix,
                 first column refers to "x" axis,
                 second column refers to "y" axis.
        """
        return np.array(list(map(list, (zip(*np.where(data == 1))))))

    @staticmethod
    def linear_regression(data):
        """
        Computes linear regression from a given x and y coordinates
        :param data: 2D numpy array of (N,2) shape containing x and y coordinates
        :return: score: (float) r-square value (from 0 to 1)
                 slope: slope of the computed regression line (where f(x) = mx + n)
                 intercept: f(0) value. The y value where the line intercepts with x = 0
        """
        x = data[:, 1].reshape((-1, 1))
        y = data[:, 0]

        model = LinearRegression().fit(x, y)
        score = model.score(x, y)
        intercept = model.intercept_
        slope = model.coef_[0]

        return {"slope": slope, "intercept": intercept, "score": score}

    def get_mask(self, data, **params):
        """
        Computes a 2D mask of the train's waterfall trace
        :param data: 2D numpy array (in one-hot-encoding)
        :param m: (float) slope of computed line (f(x) = mx + n)
        :param n: (float) f(0) value of line
        :param bw: (float) border-width
        :return: 2D numpy array
        """
        m = params['slope']
        n = params['intercept']
        bw = self.mask_width

        ys = lambda x: m * x + n + int(bw / 2)
        yd = lambda x: m * x + n - int(bw / 2)

        mask = np.zeros(data.shape)

        for x in range(mask.shape[1]):
            mask.T[x, int(-ys(x) - 1): int(-yd(x))] = 1
            mask[:, x] = mask[::-1, x]

        return mask

    @staticmethod
    def get_mask_time_indexes_range(mask_section):
        """
        From a given one-hot 2D array, computes the max upper index value and min lower index value
        :param mask_section: 2D numpy array
        :return: (tuple) (int) lower_lim, (int) upper_lim
        """
        mask_section_argmax_left_high = mask_section.shape[0] - np.argmax(mask_section[:, 0][::-1]) - 1
        mask_section_argmax_left_low = np.argmax(mask_section[:, 0])
        mask_section_argmax_right_high = mask_section.shape[0] - np.argmax(
            mask_section[:, mask_section.shape[1] - 1][::-1]) - 1
        mask_section_argmax_right_low = np.argmax(mask_section[:, mask_section.shape[1] - 1])
        upper_lim = max(mask_section_argmax_left_high, mask_section_argmax_right_high)
        lower_lim = min(mask_section_argmax_left_low, mask_section_argmax_right_low)
        return lower_lim, upper_lim

    def get_mask_subsections(self):
        """
        Deletes all empty rows and columns
        :param data: 2D numpy array
        :return: 2D numpy array
        """
        if self.mask_sections:
            spatial_section_length = self.mask_sections[0].shape[1]
            subsection_spatial_length = int(spatial_section_length / self.sub_section_num)

            for s_idx, mask_section in enumerate(self.mask_sections):
                # Get temporal index pairs
                mask_subsections = [mask_section[:, i * subsection_spatial_length:(i + 1) * subsection_spatial_length]
                                    for i
                                    in range(self.sub_section_num)]
                temporal_subsection_indexes_pairs = [self.get_mask_time_indexes_range(mask_section) for mask_section in
                                                     mask_subsections]
                logger.info(f"section - {s_idx} temporal indexes pairs: {temporal_subsection_indexes_pairs}")

                # Get spatial index pairs
                spatial_subsection_indexes = [x for x in range(0, spatial_section_length + subsection_spatial_length,
                                                               subsection_spatial_length)]
                spatial_subsection_indexes_pairs = [(spatial_subsection_indexes[i], spatial_subsection_indexes[i + 1])
                                                    for i in range(len(spatial_subsection_indexes) - 1)]
                logger.info(f"section - {s_idx} spatial indexes pairs: {spatial_subsection_indexes_pairs}")
                tsi = temporal_subsection_indexes_pairs
                ssi = spatial_subsection_indexes_pairs

                for i in range(self.sub_section_num):
                    logger.info(f"{tsi[i][0]}, {tsi[i][1]}, {ssi[i][0]}, {ssi[i][1]}")
                    self.mask_sub_sections.append(
                        self.sections[s_idx][tsi[i][0]:tsi[i][1], ssi[i][0]:ssi[i][1]]
                    )

    @staticmethod
    def auto_crop(data):
        """
        Deletes all empty rows and columns
        :param data: 2D numpy array
        :return: 2D numpy array
        """
        data = data[
               :, [any(np.array(data[:, x])) for x in range(data.shape[1])]
               ]
        return data[
               [any(np.array(data[x, :])) for x in range(data.shape[0])], :
               ]

    def filter_zeros(self, data):
        """
        Filters 0'values from a given 2D matrix columns by column (in waterfall's time-domain)
        :param data: 2D numpy array
        :return: 2D numpy array
        """
        bw = self.mask_width
        out = np.zeros([int(bw + 1), data.shape[1]])
        for x in range(data.shape[1]):
            values = data[:, x]
            filtered_values = values[values != 0]
            out[:, x] = np.pad(filtered_values, (0, int(bw + 1) - filtered_values.shape[0]), 'constant')
        return out

    def get_direction(self):
        slopes = [section['slope'] for section in self.linear_reg_params]
        self.direction = [self.positive_direction if m > 0 else self.negative_direction for m in slopes][0]

    @staticmethod
    def get_psd(data, fs=1000):
        psd_sum = []
        for x in range(0, data.shape[1], 1):
            s = data[:, x]
            (f, S) = signal.welch(s, fs, nperseg=data.shape[0])
            psd_sum.append(sum(S))
        return sum(psd_sum)

    def get_rail_id(self):
        self.rail_id = np.argmax([self.get_psd(x) for x in self.rail_view])

    def get_rail_id_confidence(self, slope_conf=0.4, pos_conf=10):
        energy_sections = [self.get_psd(x) for x in self.rail_view]
        _get_confidence: Callable[[int], int] = lambda x: 1 / (1 + np.exp(-1 * slope_conf * (x - pos_conf)))
        self.energy_diff = abs(abs(energy_sections[0]) - abs(energy_sections[1]))
        self.rail_id_confidence = round(_get_confidence(self.energy_diff), self.decimal)

    def get_speed(self, d_eff=470 / 110, decimal=3, f=3.6):
        dt = self.dt
        slopes = [x['slope'] for x in self.linear_reg_params]
        _get_speed: Callable[[int], int] = lambda slope: d_eff * f / (slope * dt)
        speeds = [abs(_get_speed(slope)) for slope in slopes]
        self.speed = round(np.mean(speeds), decimal)
        self.speed_error = round(self.speed - min(speeds), decimal)

    @staticmethod
    def normalize(a, b=2, c=1):
        return b * (a - min(a)) / (max(a) - min(a)) - c

    def get_train_track(self, start=10, end=98, offset_lim=50):
        rail_view = self.rail_view[self.rail_id][:, start:end]
        rail_view_temp = np.zeros(rail_view.shape)

        for x in range(rail_view.shape[1]):
            rail_view_temp[:, x] = self.normalize(rail_view[:, x])

        rail_view_mean = rail_view_temp.mean(axis=1)
        offset = np.median(rail_view_mean[:offset_lim])
        self.train_track = rail_view_mean - offset

    @staticmethod
    def get_confidence(val: int,
                       r: int = 2,
                       best_dist: float = 0.5,
                       decimal: int = 3,
                       method: str = "gaussian"):

        if method == "exponential":
            result = round(min(np.exp(-1 * (val - best_dist) / r), 1), decimal)
        elif method == "gaussian":
            result = round(min(np.exp(-1 * (val - best_dist) ** 2 / r ** 2), 1), decimal)
        elif method == "reciprocal":
            result = round(min(1 / (val ** r + best_dist * 1.5), 1), decimal)
        elif method == "custom":
            result = round(min((best_dist / val), 1), decimal)
        else:
            raise ValueError(f"method={method} is not supported")
        return result

    def get_train_id_confidence(self, base_train_class_data_list):
        dtw_distances_and_base_data = []
        for base_data in base_train_class_data_list:
            distance = dtw.distance_fast(self.train_track, base_data, use_pruning=True)
            dtw_distances_and_base_data.append({"distance": distance, "base-data": base_data})
        min_dtw_distances_and_base_data = min(dtw_distances_and_base_data, key=lambda x: x['distance'])
        min_dtw_distance = min_dtw_distances_and_base_data["distance"]
        best_base_data = min_dtw_distances_and_base_data["base-data"]

        return self.get_confidence(min_dtw_distance, method=self.method, decimal=self.decimal), best_base_data

    def get_train_id_info(self):
        results = []
        for data_dict in self.base_data:
            base_train_class_data_list = data_dict.get('data')
            train_id = data_dict.get('train-id')
            train_class = data_dict.get('train-class')
            confidence, best_base_data = self.get_train_id_confidence(base_train_class_data_list)
            results.append(
                {"confidence": confidence, "train-id": train_id, "train-class": train_class, "method": "gaussian",
                 "best-base-data": best_base_data})
        return max(results, key=lambda x: x['confidence'])
