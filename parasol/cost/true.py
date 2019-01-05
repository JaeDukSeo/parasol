from .common import CostFunction
import numpy as np

__all__ = ['TrueCost']

class TrueCost(CostFunction):

    def __init__(self, ds, da, network=None):
        super(TrueCost, self).__init__(ds, da)

    def get_parameters(self):
        return []

    @property
    def is_learned(self):
        return False

    def cost_fn(self, states, actions):
        pass

    def loss_fn(self, predictions, labels):
        return T.zeros(T.shape(labels))