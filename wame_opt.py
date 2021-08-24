import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import state_ops, math_ops, array_ops


#
# Adapted with reference to Tensorflow v2 tutorials:
#       https://www.tensorflow.org/tutorials/structured_data/feature_columns
#       https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
#       https://keras.io/api/optimizers/
#       https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop
# Combined with additional inspiration from:
#       https://github.com/nitbix/keras-oldfork/blob/master/keras/optimizers.py
#


class WAME(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01, name='WAME', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('eta_pos', tf.Variable(1.2))
        self._set_hyper('eta_neg', tf.Variable(0.1))
        self._set_hyper('eta_max', tf.Variable(100.0))
        self._set_hyper('eta_min', tf.Variable(0.01))
        self._set_hyper('alpha', tf.Variable(0.9))
        self.epsilon = 1e-7

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'eta_pos': self._serialize_hyperparameter('eta_pos'),
            'eta_neg': self._serialize_hyperparameter('eta_neg'),
            'eta_max': self._serialize_hyperparameter('eta_max'),
            'eta_min': self._serialize_hyperparameter('eta_min'),
            'alpha': self._serialize_hyperparameter('alpha'),
        }

    def _create_slots(self, var_list):
        [self.add_slot(var, 'grad', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'zeta', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'beta', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'zedd', initializer=tf.zeros) for var in var_list]

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(WAME, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                eta_pos=self._get_hyper('eta_pos', var_dtype),
                eta_neg=self._get_hyper('eta_neg', var_dtype),
                eta_min=self._get_hyper('eta_min', var_dtype),
                eta_max=self._get_hyper('eta_max', var_dtype),
                alpha=self._get_hyper('alpha', var_dtype),
                one_minus_alpha=1. - self._get_hyper('alpha', var_dtype)))

    def _resource_apply_dense(self, grad_t, vars, apply_state):
        # get the stored hyper parameters
        hyper = apply_state.get((vars.device, vars.dtype.base_dtype))

        # extract the stored state from the previous iteration
        grad = self.get_slot(vars, 'grad')
        beta = self.get_slot(vars, 'beta')
        zeta = self.get_slot(vars, 'zeta')
        zedd = self.get_slot(vars, 'zedd')

        # multiply current gradient by stored gradient to check the sign
        gt_g = grad_t * grad

        # set zeta for this iteration according to the gradient sign and capped eta values
        zeta_t = tf.where(tf.greater(gt_g, tf.zeros_like(gt_g)),
                          tf.minimum(zeta * hyper['eta_pos'], hyper['eta_max']),
                          tf.maximum(zeta * hyper['eta_neg'], hyper['eta_min']))

        # set Z for current iteration
        zedd_t = hyper['alpha'] * zedd + hyper['one_minus_alpha'] * zeta_t
        # set beta for current iteration
        beta_t = hyper['alpha'] * beta + hyper['one_minus_alpha'] * grad_t ** 2
        # weights update for this iteration
        vars_t = vars - (hyper['lr_t'] / zedd_t) * grad_t / (math_ops.sqrt(beta_t) + hyper['epsilon'])
        # store current state for next iteration
        grad.assign(grad_t, use_locking=self._use_locking)
        beta.assign(beta_t, use_locking=self._use_locking)
        zeta.assign(zeta_t, use_locking=self._use_locking)
        zedd.assign(zedd_t, use_locking=self._use_locking)

        # store updated weights and return
        return state_ops.assign(vars, vars_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError
