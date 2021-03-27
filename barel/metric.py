import numpy as np


class Metric(object):
    """
    A simple metric class for housing evaluations over timesteps (or a one off evaluation)

    This is intentionally decoupled from rollout to ensure we have "batch-rl"
    """

    def __init__(self, history=[]):
        self.history = history
        self.current_buffer = None
        self.summary = []
        if len(history) > 0:
            for hist in history:
                self.summary.append(self.calculate_summary(hist))

    def calculate_summary(self, hist):
        summary = {
            "max": np.max(hist),
            "min": np.min(hist),
            "mean": np.mean(hist),
            "std": np.std(hist),
            "10p": np.percentile(hist, 10),
            "25p": np.percentile(hist, 25),
            "50p": np.percentile(hist, 50),
            "75p": np.percentile(hist, 75),
            "90p": np.percentile(hist, 90),
        }
        return summary

    def add(self, run):
        if self.current_buffer is None:
            self.current_buffer = [run]
        else:
            self.current_buffer.append(run)
        return self

    def update(self, run=None):
        if run is not None:
            self.add(run)

        hist = self.current_buffer.copy()
        self.history.append(hist)
        self.summary.append(self.calculate_summary(hist))
        self.current_buffer = []
        return self
