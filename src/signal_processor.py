import numpy as np
from scipy import signal

# -------------------------------------------------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler()
# handler = logging.FileHandler('main.log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
# -----------------------------------------------------------------------------------------------------------------


class SignalProcessor:
    def __init__(self, data: np.ndarray, **config):
        self.data = data
        self.debug = config['debug']
        self.N = config['signal']['N']
        self.f_order = config['signal']['f_order']
        self.Wn = config['signal']['Wn-mask']
        self.btype = config['signal']['btype']
        self.fs = config['signal']['fs']
        self.dt = (1 / self.fs) * self.N
        self.temporal_length = self.data.shape[0]
        self.spatial_length = self.data.shape[1]

        self.reduced_data = self.movmean_and_downsample()
        self.filtered_data = self.butterworth_filter()

        if self.debug:
            logger.debug(f"Input Data Size: {self.data.shape}")
            logger.debug(f"Reduced Data to Size: {self.reduced_data.shape}")

    def movmean_and_downsample(self):
        """
        Function to apply a moving average (lowpass filter) followed by
        a downsampling operation.

        Parameters
        ---------- 
        data   : ndarray
            Data in 2D matrix.
        N      : int
            Number of temporal samples to average. 

        Returns
        -------
        reduced_data: 
            data after filtering and size reduction.  

        Notes
        -----
        - It is assumed the matrix `data` has a structur of (time samples, spatio indeces).
        - Each signal is assumed to be in a column of the matrix s. 
        """

        new_temporal_length = len(np.arange(0, self.temporal_length, self.N))
        reduced_data = np.empty([new_temporal_length, self.spatial_length])

        count = 0
        for i in range(0, self.temporal_length, self.N):
            start = i
            end = min(i + self.N, self.temporal_length)
            reduced_data[count, :] = np.mean(self.data[start:end, :], axis=0)
            count = count + 1

        return reduced_data

    def butterworth_filter(self):
        """
        Code to apply a zero-phase filtering to signal.
        Each signal is assumed to be in a column of the matrix s.

        Parameters
        ----------
        data       : np.array()
            Data matrix.
        dt         : int
            Sampling period of signals [s].
        fc         :
            Filter cutoff frequency [Hz]. Single value is required
            for low and high pass filter. For bandpass filter a list
            of two values is required [f_low,f_high].

        filter_type: str
            String containing filter type: "hp","lp","bp"
        :return:
        """

        fs = 1 / self.dt

        # Calculate filter coefficients:
        b, a = signal.butter(
            N=self.f_order, Wn=self.Wn, btype=self.btype, fs=fs
        )  # TODO: In signal.butter, N parameter is filter's order!

        # Apply filter to signal:
        filtered_data = signal.filtfilt(b, a, self.reduced_data, axis=0)

        return filtered_data
