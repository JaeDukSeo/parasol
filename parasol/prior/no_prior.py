from deepx import T

from .common import Prior

__all__ = ['NoPrior']

class NoPrior(Prior):

    def get_parameters(self):
        return []

    def kl_gradients(self, q_X, q_A, kl, num_data):
        return []

    def kl_divergence(self, q_X, q_A, num_data):
        batch_size = T.shape(q_X.expected_value())[0]
        return T.zeros(batch_size), {}

    def has_dynamics(self):
        return False
