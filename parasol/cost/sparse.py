import numpy as np
from deepx import T, stats

from .nn import NNCost

__all__ = ['SparseCost']

class SparseCost(NNCost):

    def cost_fn(self, states, actions):
        shape = T.shape(states)
        N, H = shape[0], shape[1]
        x = T.reshape(states, [N * H, -1])
        return T.reshape(self.network(x), [N,H])

    def loss_fn(self, predictions, labels):
       # predict = self.cost(states, actions)
       return T.mean(-(1 - labels)*T.log(1e-7+1-predictions) - labels*T.log(1e-7+predictions))