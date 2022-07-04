import argparse
import os
from datetime import datetime

import tensorflow as tf

import losses
from callbacks import CustomCallback
from dataloader import load_data
from models.haarnet import HaarNet


def main(args):
    if not os.path.exists(args.dir_logs):
        os.makedirs(args.dir_logs)
    with open(os.path.join(args.dir_logs, 'params.txt'), 'w') as file:
        params = vars(args)
        for key in params.keys():
            file.write('{}: {}\n'.format(key, params[key]))

    # Datasets
    data_train, data_val, data_test = load_data(ds_train=args.data_train,
                                                ds_val=args.data_val,
                                                ds_test=args.data_test,
                                                sz_trn=args.size_train,
                                                batch_size=args.batch_size)
    # Define the Optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.adam_b1, beta_2=args.adam_b2)
    # Define loss
    loss_obj = losses.NetLoss(args.loss_mu1, args.loss_mu2)
    # Define model
    model = HaarNet(args.filters)
    # Define the Checkpoint-saver
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Train
    model.compile(optimizer=optimizer,
                  loss_fn=loss_obj)
    if args.train:
        model.fit(x=data_train, epochs=args.epochs,
                  callbacks=[CustomCallback(args.dir_logs, ckpt)],
                  validation_data=data_val)
    if args.test:
        model.evaluate(x=data_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dehazing model')
    parser.add_argument('--train', type=int, default=1, help='Enable/disable training')
    parser.add_argument('--test', type=int, default=0, help='Enable/disable testing')
    parser.add_argument('--data_train', type=str, default='nyudv2',
                        help='The directory of the train dataset - nyudv2 / diode')
    parser.add_argument('--data_val', type=str, default='middlebury',
                        help='The directory of the validation dataset')
    parser.add_argument('--data_test', type=str, default='nh_haze',
                        help='The directory of the test dataset')
    parser.add_argument('--size_train', type=int, default=5000,
                        help='The size of the train dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Set the batch size')
    parser.add_argument('--filters', type=int, default=32,
                        help='The number of filters in level 0 of model.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--adam_b1', type=float, default=0.5,
                        help='Beta1 decay rate for Adam.')
    parser.add_argument('--adam_b2', type=float, default=0.999,
                        help='Beta2 decay rate for Adam.')
    parser.add_argument('--loss_mu1', type=float, default=0.01, help='Set the mu1 value of loss')
    parser.add_argument('--loss_mu2', type=float, default=0.1, help='Set the mu2 value of loss')
    parser.add_argument('--epochs', type=int, default=2, help='Set the number of epochs')
    parser.add_argument('--dir_logs', type=str, default='logs/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='DO NOT MODIFY. The directory of tensorboard logs.')

    arguments = parser.parse_args()

    tic = datetime.now()
    main(arguments)
    toc = datetime.now()
    print('Program started at', tic)
    print('Program ended at', toc)
    print('Total time taken', toc - tic)
