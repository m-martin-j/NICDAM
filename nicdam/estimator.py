
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class AbstractEstimator(ABC):
    """Defines abstract properties of estimators."""

    def __init__(self) -> None:
        pass

    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class Estimator(AbstractEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass

    def predict(self):
        pass


class IncrementalEstimator(AbstractEstimator):

    def fit_incremental(self):
        pass


class Regressor(Estimator):
    pass


class Classifier(Estimator):
    pass
