from abc import ABC
from functools import wraps


def rank_zero_only(fn):
    """Decorate a logger method to run it only on the process with rank 0.

    :param fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn


class LightningLoggerBase(ABC):
    """Base class for experiment loggers."""

    def __init__(self):
        self._rank = 0

    @property
    def experiment(self):
        raise NotImplementedError()

    def log_metrics(self, metrics, step):
        """Record metrics.

        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError()

    def log_hyperparams(self, params):
        """Record hyperparameters.

        :param params: argparse.Namespace containing the hyperparameters
        """
        raise NotImplementedError()

    def save(self):
        """Save log data."""
        pass

    def finalize(self, status):
        """Do any processing that is necessary to finalize an experiment.

        :param status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        pass

    def close(self):
        """Do any cleanup that is necessary to close an experiment."""
        pass

    @property
    def rank(self):
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value):
        """Set the process rank."""
        self._rank = value

    @property
    def name(self):
        """Return the experiment name."""
        raise NotImplementedError("Sub-classes must provide a name property")

    @property
    def version(self):
        """Return the experiment version."""
        raise NotImplementedError("Sub-classes must provide a version property")
