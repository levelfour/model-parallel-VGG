from __future__ import print_function

import argparse

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainermn


import VGG


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Create ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        device = comm.rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(VGG.VGG(comm, 10))

    if args.gpu:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Load the CIFAR10 dataset
    train, test = chainer.datasets.get_cifar10()
    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        test = chainermn.datasets.create_empty_dataset(test)

    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(train, args.batchsize), comm)
    test_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(test, args.batchsize,
                                         repeat=False, shuffle=False),
        comm)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    if comm.rank == 0:
        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))
    
        # Take a snapshot for each specified epoch
        # TODO: deadlock
#        frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
#        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
    
        # Save two plot images to the result dir
        if args.plot and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))
    
        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
