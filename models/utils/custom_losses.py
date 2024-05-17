import tensorflow as tf


class UncertaintyLoss(tf.keras.layers.Layer):
    '''
    The adaptive loss
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    '''

    def __init__(self, loss_keys, **kwargs):
        self.is_placeholder = True
        self.loss_keys = loss_keys
        super(UncertaintyLoss, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        # These are the log(sigma ** 2) parameters.
        self.log_vars = []
        for key in self.loss_keys:
            self.log_vars += [
                self.add_weight(name=str(key) + '_' + 'var',
                                shape=(1, ),
                                initializer=tf.keras.initializers.Constant(0.),
                                trainable=True)
            ]
        super(UncertaintyLoss, self).build(input_shape)

    def call(self, losses):
        total_loss = 0
        for key, log_var in zip(self.loss_keys, self.log_vars):
            var = log_var[0]
            precision = tf.math.exp(-var)
            loss = tf.math.reduce_sum(.5 * precision * losses[key] + .5 * var)
            total_loss += loss
        return total_loss


class CoVWeightingLoss(tf.keras.layers.Layer):
    '''
    The adaptive loss
    MULTI-LOSS WEIGHTING WITH COEFFICIENT OF VARIATIONS
    https://arxiv.org/abs/2009.01717
    '''

    def __init__(self, loss_keys, mean_decay_param=None):
        self.is_placeholder = True
        # use mean decay for more robust from paper section 2.1
        if mean_decay_param is not None and mean_decay_param != 0.:
            self.mean_decay_param = mean_decay_param
        self.loss_keys = loss_keys
        self.num_losses = len(self.loss_keys)
        self.current_iter = -1
        self.running_std_l = None
        super(CoVWeightingLoss, self).__init__()

    def build(self, input_shape=None):
        # Initialize all running statistics at 0.
        self.alphas = self.add_weight(
            name='alpha',
            shape=(self.num_losses, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=False)

        self.running_mean_L = self.add_weight(
            name='running_mean_L',
            shape=(self.num_losses, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=False)

        self.running_mean_l = self.add_weight(
            name='running_mean_l',
            shape=(self.num_losses, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=False)

        self.running_S_l = self.add_weight(
            name='running_S_l',
            shape=(self.num_losses, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=False)
        self.alphas = tf.zeros(
            (self.num_losses, ),
            requires_grad=False).type(torch.FloatTensor).to(self.device)

        super(CoVWeightingLoss, self).build(input_shape)

    def call(self, unweighted_losses, training=None):
        total_loss = 0
        # input is dict unweighted_losses
        # parse to tensor
        unweighted_losses = tf.cast(
            [unweighted_losses for k in unweighted_losses], tf.float32)
        if training is False:
            for k in unweighted_losses:
                total_loss += unweighted_losses[k]
            return total_loss
        # Increase the current iteration parameter.
        self.current_iter += 1

        unweighted_losses = unweighted_losses if self.current_iter == 0 else self.running_mean_L

        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = tf.ones(self.num_losses, ) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / tf.mtah.reduce_sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l

        x_l = tf.stop_gradient(l)
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = tf.stop_gradient(L)
        self.running_mean_L = mean_param * \
            self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [
            self.alphas[i] * unweighted_losses[i]
            for i in range(len(unweighted_losses))
        ]
        loss = sum(weighted_losses)
        return total_loss
