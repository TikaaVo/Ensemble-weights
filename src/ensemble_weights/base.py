from abc import ABC, abstractmethod


class BaseRouter(ABC):
    @abstractmethod
    def fit(self, features, y, preds_dict):
        pass

    @abstractmethod
    def predict(self, x):
        pass