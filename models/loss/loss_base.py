import abc


class LossBase(abc.ABC):
    @abc.abstractmethod
    def build_loss(self, logit_lists, targets, is_train):
        raise NotImplementedError
