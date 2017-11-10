from keras import backend as K
# import numpy as np
from six.moves import zip
from keras.optimizers import Optimizer
from keras.legacy import interfaces


class Adam_accumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, accum_iters=4, decay=0., **kwargs):
        super(Adam_accumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(
                accum_iters, name='accum_iters')
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = ms + vs

        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            flag = K.equal(self.iterations % self.accum_iters, 0)
            flag = K.cast(flag, dtype='float32')
            # print(self.accum_iters)

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * \
                (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * \
                K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, flag * m_t + (1 - flag) * m))
            self.updates.append(K.update(v, flag * v_t + (1 - flag) * v))
            self.updates.append(K.update(gg, gg_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adam_accumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
