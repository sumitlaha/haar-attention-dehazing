import tensorflow as tf

from models.layers import MainModule
from metrics import PSNR, SSIM

loss_tracker = tf.keras.metrics.Mean(name='loss')
psnr_metric = PSNR(name='psnr')
ssim_metric = SSIM(name='ssim')


class HaarNet(tf.keras.Model):
    def __init__(self, filters):
        super(HaarNet, self).__init__()
        self.net = MainModule(filters)

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training=training)

    def compile(self, optimizer, loss_fn, run_eagerly=None):
        super(HaarNet, self).compile()
        self.optimizer = optimizer
        self.loss_fn1 = loss_fn
        self.run_eagerly = run_eagerly


    @tf.function
    def train_step(self, data):
        im_i, im_j = data
        with tf.GradientTape() as tape:
            # Predictions
            yp = self.net(im_i, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_net1 = self.loss_fn1(im_j, yp)
        # Compute gradients
        grads_net1 = tape.gradient(loss_net1, self.net.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(grads_net1, self.net.trainable_variables))

        # Compute metrics
        loss_tracker.update_state(loss_net1)
        # Update metrics (includes the metric that tracks the loss)
        psnr_metric.update_state(im_j, yp)
        ssim_metric.update_state(im_j, yp)
        return {'loss': loss_tracker.result(), 'psnr': psnr_metric.result(), 'ssim': ssim_metric.result()}

    @tf.function
    def test_step(self, data):
        im_i, im_j = data
        # Compute predictions
        yp = self.net(im_i, training=True)
        # Updates the metrics tracking the loss

        loss_net1 = self.loss_fn1(im_j, yp)

        # Update the metrics.
        loss_tracker.update_state(loss_net1)
        psnr_metric.update_state(im_j, yp)
        ssim_metric.update_state(im_j, yp)
        return {'loss': loss_tracker.result(), 'psnr': psnr_metric.result(), 'ssim': ssim_metric.result()}

    # @property
    # def metrics(self):
    #     return [loss_tracker, psnr_metric, ssim_metric]
