import numpy as np
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DataPlotter:
    def __init__(self, data, **config):
        self.data = data
        self.vmin = config['vmin']
        self.vmax = config['vmax']
        self.xlabel = config['xlabel']
        self.ylabel = config['ylabel']
        self.title = config['title']
        self.cmap = config['cmap']
        self.figsize = config['figsize']
        self.extent = config['extent']
        self.fig = None

    def plot_matrix(self):
        # ---- Set default colorbar limits:
        # vmin and vmax corresponds to the 2nd and 98th percentile of the data:
        if self.vmin is None:
            self.vmin = np.percentile(self.data, 2)
        if self.vmax is None:
            self.vmax = np.percentile(self.data, 98)

        # ---- Set default figure size:
        if self.figsize is None:
            self.figsize = (19.2, 9.83)

        # ---- Plot figure:
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        img = ax.imshow(
            self.data,
            aspect="auto",
            origin="lower",
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="none",
            extent=self.extent,
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        fig.colorbar(img)
        fig.tight_layout()
        plt.show()

        self.fig = fig
        return None

    def plot_train_track(self):
        figsize = (18, 9)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
