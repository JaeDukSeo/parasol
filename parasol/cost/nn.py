import pickle
from deepx import T, nn, stats
from .common import CostFunction
import numpy as np

__all__ = ['NNCost']

class NNCost(CostFunction):

    def __init__(self, ds, da, network=None):
        super(NNCost, self).__init__(ds, da)
        assert network is not None, 'Must specify network'
        self.architecture = pickle.dumps(network)
        self.network = network

    def __getstate__(self):
        state = super(NNCost, self).__getstate__()
        state['architecture'] = self.architecture
        state['weights'] = T.get_current_session().run(self.get_parameters())
        return state

    def __setstate__(self, state):
        network = pickle.loads(state.pop('architecture'))
        weights = state.pop('weights')
        self.__init__(state['ds'], state['da'], network=network)
        T.get_current_session().run([T.core.assign(a, b) for a, b in zip(self.get_parameters(), weights)])

    def get_parameters(self):
        return self.network.get_parameters()

    @property
    def is_learned(self):
        return True

    def cost_fn(self, states, actions):
        x = T.concatenate([states, actions], -1)
        shape = T.shape(x)
        N, H = shape[0], shape[1]
        x = T.reshape(x, [N * H, -1])
        return T.reshape(self.network(x), [N,H])

    def loss_fn(self, predictions, labels):
        # predict = self.cost(states, actions)
        return T.mean(T.square(predictions - labels))
