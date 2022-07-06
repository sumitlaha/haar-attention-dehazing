import tensorflow as tf

from models.layers import MainModule
from metrics import PSNR, SSIM

loss_train = tf.keras.metrics.Mean(name='train_loss')
loss_test = tf.keras.metrics.Mean(name='test_loss')
psnr_metric = PSNR(name='psnr')
ssim_metric = SSIM(name='ssim')


class HaarNet(tf.keras.Model):
    def __init__(self, filters):
        super(HaarNet, self).__init__()
        self.net = MainModule(filters)
        self.optimizer = None
        self.loss_fn = None

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training=training)

    def compile(self, optimizer, loss_fn):
        super(HaarNet, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):
        im_i, im_j = data
        with tf.GradientTape() as tape:
            # Compute predictions
            yp = self.net(im_i, training=True)
            # Compute loss
            loss_net = self.loss_fn(im_j, yp)
        # Compute gradients
        grads_net = tape.gradient(loss_net, self.net.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(grads_net, self.net.trainable_variables))
        # Compute metrics
        loss_train.update_state(loss_net)
        # Update metrics (includes the metric that tracks the loss)
        psnr_metric.update_state(im_j, yp)
        ssim_metric.update_state(im_j, yp)
        return {'loss': loss_train.result(), 'psnr': psnr_metric.result(), 'ssim': ssim_metric.result()}

    @tf.function
    def test_step(self, data):
        im_i, im_j = data
        # Compute predictions
        yp = self.net(im_i, training=True)
        # Updates metrics tracking loss
        loss_net = self.loss_fn(im_j, yp)
        # Update metrics
        loss_test.update_state(loss_net)
        psnr_metric.update_state(im_j, yp)
        ssim_metric.update_state(im_j, yp)
        return {'loss': loss_test.result(), 'psnr': psnr_metric.result(), 'ssim': ssim_metric.result()}
