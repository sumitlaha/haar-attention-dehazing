import argparse
import os
from datetime import datetime

import tensorflow as tf

import losses
from callbacks import CustomCallback
from data_loader3 import load_data
from models.model18 import HaarNet


def main(args):
    if not os.path.exists(args.dir_logs):
        os.makedirs(args.dir_logs)
    with open(os.path.join(args.dir_logs, 'params.txt'), 'w') as file:
        params = vars(args)
        for key in params.keys():
            file.write('{}: {}\n'.format(key, params[key]))

    # Datasets
    ds_trn, ds_val = load_data(ds_trn=args.data_train,
                               ds_val=args.data_val,
                               ds_tst=args.data_test,
                               sz_trn=args.size_train,
                               batch_size=args.batch_size,
                               aug=args.aug)
    # Define the Optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.6, beta_2=0.999)
    # Define loss
    # loss_obj = losses.netLoss(mu=args.mu, reduction=tf.keras.losses.Reduction.NONE)
    loss_obj = losses.netLoss(mu=args.mu, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    # Define model
    model = HaarNet(args.filters)
    # Define the Checkpoint-saver
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Train
    model.compile(optimizer=optimizer,
                  loss_fn=loss_obj)

    model.fit(x=ds_trn, epochs=args.epochs,
              callbacks=[CustomCallback(args.dir_logs, ckpt, ds_val, sample_size=5)],
              validation_data=ds_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dehazing model')
    parser.add_argument('--data_train', type=str, default='nyudv2',
                        help='The directory of the train dataset - nyudv2 / diode')
    parser.add_argument('--size_train', type=int, default=5,
                        help='The size of the train dataset')
    parser.add_argument('--data_val', type=str, default='middlebury',
                        help='The directory of the validation dataset')
    parser.add_argument('--data_test', type=str, default='sots',
                        help='The directory of the test dataset')
    parser.add_argument('--filters', type=int, default=32,
                        help='The number of filters in level 0 of model.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--aug', type=int, default=0, choices=[1, 0],
                        help='Data augmentation during training.')
    parser.add_argument('--batch_size', type=int, default=8, help='Set the batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Set the number of epochs')
    parser.add_argument('--mu', type=float, default=0.01, help='Set the mu value of loss')
    parser.add_argument('--dir_logs', type=str, default='logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='DO NOT MODIFY. The directory of tensorboard logs.')
    # parser.add_argument('--buffer_size', type=int, default=400, help='Set the buffer size')
    # parser.add_argument('--img_width', type=int, default=512, help='Set the image width')
    # parser.add_argument('--img_height', type=int, default=384, help='Set the image height')

    arguments = parser.parse_args()

    tic = datetime.now()
    main(arguments)
    toc = datetime.now()
    print('Program started at', tic)
    print('Program ended at', toc)
    print('Total time taken', toc - tic)
