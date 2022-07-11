import tensorflow as tf

from utils import display_images, standardize


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, dir_logs, ckpt, ds_test=None):
        super(CustomCallback, self).__init__()
        self.log_dir = dir_logs
        self.checkpoint = ckpt
        self.ds_test = ds_test
        self.summary_writer_trn = tf.summary.create_file_writer(dir_logs + "/train/")
        self.summary_writer_val = tf.summary.create_file_writer(dir_logs + "/val/")

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer_trn.as_default():
            tf.summary.scalar('loss', logs['loss'], step=epoch)
            tf.summary.scalar('psnr', logs['psnr'], step=epoch)
            tf.summary.scalar('ssim', logs['ssim'], step=epoch)
        with self.summary_writer_val.as_default():
            tf.summary.scalar('loss', logs['val_loss'], step=epoch)
            tf.summary.scalar('psnr', logs['val_psnr'], step=epoch)
            tf.summary.scalar('ssim', logs['val_ssim'], step=epoch)

        # saving (checkpoint) the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            self.checkpoint.save(file_prefix=self.log_dir + '/tf_ckpts/ckpt')
            print('\nCheckpoint saved...\n')

    def on_test_end(self, logs=None):
        if self.ds_test is not None:
            batch_idx = 0
            titles = ['i', 'yp', 'j']
            imgs = []
            for im_i, im_j in self.ds_test.take(5):
                # Compute predictions
                yp = self.model.predict(im_i)
                yp = standardize(yp, [1, 2])
                imgs.extend([im_i[batch_idx], yp[batch_idx], im_j[batch_idx]])
            display_images(imgs, titles, 5)
