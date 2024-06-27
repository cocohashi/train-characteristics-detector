import numpy as np
import matplotlib.pyplot as plt

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


class DataPlotter:
    def __init__(self, data, **config):
        self.data = data
        self.vmin = config.get('vmin')
        self.vmax = config.get('vmax')
        self.xlabel = config.get('xlabel')
        self.ylabel = config.get('ylabel')
        self.title = config.get('title')
        self.cmap = config.get('cmap')
        self.figsize = config.get('figsize')
        self.extent = config.get('extent')
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

    def plot_train_track(self):

        # ---- Set default figure size:
        if self.figsize is None:
            self.figsize = (19.2, 9.83)

        train_track = self.data[0]
        best_base_data = self.data[1]

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.plot(train_track, label=f"Train Track")
        ax.plot(best_base_data, label="Best Base Train Track")
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.legend(loc="upper right")
        plt.show()

        self.fig = fig
