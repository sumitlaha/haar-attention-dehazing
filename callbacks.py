import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, dir_logs, ckpt):
        super(CustomCallback, self).__init__()
        self.log_dir = dir_logs
        self.checkpoint = ckpt
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
