# import tensorflow as tf
# import numpy as np
# from .common import CostFunction
# from .util import create_cost_matrix

# class QuadraticCostFunction(CostFunction):

#     def _initialize(self, *params, placeholder=False):
#         D = params[0]
#         if not placeholder:
#             if self.tv:
#                 mat, opt_params = create_cost_matrix(D, psd=False, tv=True, episode_length=self.episode_length)
#                 cost_vector = tf.Variable(tf.zeros([self.episode_length, D]))
#                 return mat + [cost_vector], opt_params + [cost_vector]
#             else:
#                 mat, opt_params = create_cost_matrix(D, psd=False)
#                 cost_vector = tf.Variable(tf.zeros([D]))
#                 return mat + [cost_vector], opt_params + [cost_vector]
#         else:
#             if self.tv:
#                 return ([tf.placeholder(tf.float32, [self.episode_length, D, D]),
#                          tf.placeholder(tf.float32, [self.episode_length, D])], [])
#             else:
#                 return ([tf.placeholder(tf.float32, [D, D]), tf.placeholder(tf.float32, [D])], [])

#     def _cost(self, state):
#         C, c = self.params
#         if len(state.get_shape()) == 1:
#             return 0.5 * tf.einsum('a,ab,b->', state, C, state) + tf.einsum('a,a->', state, c)
#         if len(state.get_shape()) == 2:
#             if self.tv:
#                 return 0.5 * tf.einsum('ja,jab,jb->j', state, C, state) + tf.einsum('ja,ja->j', state, c)
#             else:
#                 return 0.5 * tf.einsum('ia,ab,ib->i', state, C, state) + tf.einsum('ia,a->i', state, c)
#         if len(state.get_shape()) == 3:
#             if self.tv:
#                 return 0.5 * tf.einsum('ija,jab,ijb->ij', state, C, state) + tf.einsum('ija,ja->ij', state, c)
#             else:
#                 return 0.5 * tf.einsum('ija,ab,ijb->ij', state, C, state) + tf.einsum('ija,a->ij', state, c)
