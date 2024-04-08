import logging
from typing import Callable, Any

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from scipy import signal

from dtaidistance import dtw

from .signal_processor import SignalProcessor

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class CharDetector(SignalProcessor):
    def __init__(self, signal_processor, **config):
        super(SignalProcessor, self).__init__()

        self.signal_processor = signal_processor
        self.filtered_data = signal_processor.filtered_data  # Cut-off freq: Wn-mask
        self.N = signal_processor.N
        self.dt = signal_processor.dt

        signal_processor.Wn = config['signal']['Wn-class']  # Change Cut-off freq: Wn-class
        self.section_limit = config['char-detector']['section-limit']
        self.section_num = config['char-detector']['section-num']
        self.mask_width = config['char-detector']['mask-width'] / self.N
        self.thr_perc = config['char-detector']['thr-perc']
        self.mf_window = config['char-detector']['mf-window']
        self.no_train_event_thr = config['char-detector']['no-train-event-thr']
        self.method = config['char-detector']['method']
        self.positive_direction = config['schema']['positive-direction']
        self.negative_direction = config['schema']['negative-direction']
        self.event = config['event']['train']
        self.no_train_event = config['event']['no-train']

        self.direction = ""
        self.rail_id = None
        self.speed = 0
        self.speed_error = 0
        self.train_track = []
        self.base_data = None
        self.rail_view = None

        # Slice waterfall in different spatial sections
        self.sections = self.get_sections()

        # Compute sobel and threshold filters to each section
        self.sobel_sections = [self.sobel_filter(data) for data in self.sections]
        self.thr_sections = [self.thresholding(data) for data in self.sobel_sections]

        # Get mask of each section
        self.mean_filter_sections = [self.mean_filter(data) for data in self.thr_sections]

        # Check 'NO TRAIN' event
        if self.check_event(event=self.no_train_event):
            self.event = self.no_train_event

        else:
            self.coordinates_sections = [self.one_hot_to_coordinates(data) for data in self.mean_filter_sections]
            self.linear_reg_params = [self.linear_regression(coordinate) for coordinate in self.coordinates_sections]
            self.mask_sections = [self.get_mask(self.thr_sections[x], **self.linear_reg_params[x]) for x in
                                  range(self.section_num)]

            # Filter data again using the filter defined in Wn-class
            self.filtered_data = signal_processor.butterworth_filter()  # Apply Cut-off freq: Wn-class
            self.sections = self.get_sections()

            # Apply mask to each section
            self.masked_sections = [np.multiply(self.sections[x], self.mask_sections[x]) for x in
                                    range(self.section_num)]
            self.masked_sections = [self.auto_crop(data) for data in self.masked_sections]
            self.rail_view = [self.filter_zeros(data) for data in self.masked_sections]

            self.get_direction()
            self.get_rail_id()
            self.get_speed()
            self.get_train_track()

    def get_sections(self):
        """
        Subdivides matrix into "n" Spatial Sections by a given "section limit"
        :return: 2D Numpy array
        """
        s_lim = self.section_limit
        n = self.section_num
        return [self.filtered_data[:, x * s_lim: (x + 1) * s_lim] for x in range(n)]

    def sobel_filter(self, data, sigma=1):
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
        column_filter = ndimage.median_filter(data, size=self.mf_window, axes=1)
        return ndimage.median_filter(column_filter, size=self.mf_window, axes=0)

    def check_event(self, event):
        if event == self.no_train_event:
            mean = np.mean([np.mean(self.mean_filter_sections[x]) for x in range(self.section_num)])
            return bool(mean < self.no_train_event_thr)

    def one_hot_to_coordinates(self, data):
        """
        Obtains x and y coordinate axis from one-hot-encoded 2D array matrix
        :param data: 2D numpy array (in one-hot-encoding format)
        :return: 2D numpy array of shape (N,2) where N depends on the number of "1's" in the matrix,
                 first column refers to "x" axis,
                 second column refers to "y" axis.
        """
        return np.array(list(map(list, (zip(*np.where(data == 1))))))

    def linear_regression(self, data):
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

    def auto_crop(self, data):
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

    def get_speed(self, d_eff=470 / 110, dec=3, f=3.6):
        dt = self.dt
        slopes = [x['slope'] for x in self.linear_reg_params]
        _get_speed: Callable[[int], int] = lambda slope: round(d_eff * f / (slope * dt), dec)
        speeds = [abs(_get_speed(slope)) for slope in slopes]
        self.speed = np.mean(speeds)
        self.speed_error = round(self.speed - min(speeds), dec)

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
                       decimal: int = 5,
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

    def get_train_id_confidence(self, base_train_ids):
        dtw_distances = []
        for base_train_id in base_train_ids:
            distance = dtw.distance_fast(self.train_track, base_train_id, use_pruning=True)
            dtw_distances.append(distance)
        return self.get_confidence(np.mean(dtw_distances), method=self.method)

    def get_train_id_info(self):
        results = []
        for data_dict in self.base_data:
            base_train_ids = data_dict.get('data')
            train_id = data_dict.get('train-id')
            train_class = data_dict.get('train-class')
            confidence = self.get_train_id_confidence(base_train_ids)
            results.append(
                {"confidence": confidence, "train-id": train_id, "train-class": train_class, "method": "gaussian"})
        return max(results, key=lambda x: x['confidence'])
