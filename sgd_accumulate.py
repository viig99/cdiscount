from keras import backend as K
# import numpy as np
from six.moves import zip
from keras.optimizers import Optimizer
from keras.legacy import interfaces

class SGDAccum(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accum_iters=1, **kwargs):
        super(SGDAccum, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(accum_iters)
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        accum_switch = K.equal(self.iterations % self.accum_iters, 0)
        accum_switch = K.cast(accum_switch, dtype='float32')

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        temp_grads = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, cg, m, tg in zip(params, grads, moments, temp_grads):
            g = cg + tg
            v = self.momentum * m - (lr * g / self.accum_iters)  # velocity
            self.updates.append(K.update(m, (1 - accum_switch) * m + accum_switch * v))
            self.updates.append(K.update(tg, (1 - accum_switch) * g))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * g / self.accum_iters)
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - accum_switch) * p + accum_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'accum_iters': self.accum_iters}
        base_config = super(SGDAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))