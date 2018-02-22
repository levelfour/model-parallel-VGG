from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainermn.functions


def debug(*s):
    import sys
    from mpi4py.MPI import COMM_WORLD
    print('[rank:{}]'.format(COMM_WORLD.Get_rank()), *s, file=sys.stderr, flush=True)


class ParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, device, rank_root, in_channels, *args, **kwargs):
        assert in_channels == 3, "actual in_channels is {}".format(in_channels)  # TODO enable to split channels evenly
        # Set in_channels = 1.
        super(ParallelConvolution2D, self).__init__(1, *args, **kwargs)
        self.comm = comm
        self.device = device
        self.rank_root = rank_root

    def __call__(self, *inputs):
        if self.comm.rank == self.rank_root:
            x, = inputs
            phi = None

            # scatter
            for rank in range(self.comm.size):
                if rank == self.comm.rank:
                    _x = x[:, rank:rank+1]
                else:
                    _phi = chainermn.functions.send(x[:, rank:rank+1], self.comm, rank)
                    if phi is not None:
                        phi = chainermn.functions.pseudo_connect(phi, _phi)
                    else:
                        phi = _phi

            # convolution
            y = super(ParallelConvolution2D, self).__call__(_x)

            # gather -> reduce
            for rank in range(self.comm.size):
                if rank != self.comm.rank:
                    # NOTE: every recv needs phi, to maintain backward priority
                    _y = chainermn.functions.recv(self.comm, rank, device=self.device, delegate_variable=phi)
                    y += _y

            return y

        else:
            if len(inputs) > 0:
                phi, = inputs
            else:
                phi = None
            x = chainermn.functions.recv(self.comm, self.rank_root, device=self.device, delegate_variable=phi)
            y = super(ParallelConvolution2D, self).__call__(x)
            return chainermn.functions.send(y, self.comm, self.rank_root)


class MasterBlock(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, comm, device, out_channels, ksize, pad=1):
        super(MasterBlock, self).__init__()
        with self.init_scope():
            self.conv = ParallelConvolution2D(comm, device, 0, 3, out_channels, ksize, pad=pad, nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class SlaveBlock(chainer.Chain):
    def __init__(self, comm, device, out_channels, ksize, pad=1):
        super(SlaveBlock, self).__init__()
        with self.init_scope():
            self.conv = ParallelConvolution2D(comm, device, 0, 3, out_channels, ksize, pad=pad, nobias=True)

    def __call__(self, *inputs):
        # Passing delegate variables.
        return self.conv(*inputs)


class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, comm, device, class_labels=10):
        super(VGG, self).__init__()
        self.comm = comm
        if comm.rank == 0:  # master
            Block = MasterBlock
        else:  # slave
            Block = SlaveBlock

        with self.init_scope():
            self.block1_1 = Block(comm, device, 3, 3) # Block(64, 3)
            self.block1_2 = Block(comm, device, 3, 3) # Block(64, 3)
            self.block2_1 = Block(comm, device, 3, 3) # Block(128, 3)
            self.block2_2 = Block(comm, device, 3, 3) # Block(128, 3)
            self.block3_1 = Block(comm, device, 3, 3) # Block(256, 3)
            self.block3_2 = Block(comm, device, 3, 3) # Block(256, 3)
            self.block3_3 = Block(comm, device, 3, 3) # Block(256, 3)
            self.block4_1 = Block(comm, device, 3, 3) # Block(512, 3)
            self.block4_2 = Block(comm, device, 3, 3) # Block(512, 3)
            self.block4_3 = Block(comm, device, 3, 3) # Block(512, 3)
            self.block5_1 = Block(comm, device, 3, 3) # Block(512, 3)
            self.block5_2 = Block(comm, device, 3, 3) # Block(512, 3)
            self.block5_3 = Block(comm, device, 3, 3) # Block(512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, *inputs):
        debug('call')
        if self.comm.rank == 0:  # master
            x, = inputs

            # 64 channel blocks:
            h = self.block1_1(x)
            h = F.dropout(h, ratio=0.3)
            h = self.block1_2(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
    
            # 128 channel blocks:
            h = self.block2_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block2_2(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
    
            # 256 channel blocks:
            h = self.block3_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block3_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block3_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
    
            # 512 channel blocks:
            h = self.block4_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block4_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block4_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
    
            # 512 channel blocks:
            h = self.block5_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block5_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block5_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
    
            h = F.dropout(h, ratio=0.5)
            h = self.fc1(h)
            h = self.bn_fc1(h)
            h = F.relu(h)
            h = F.dropout(h, ratio=0.5)
            h = self.fc2(h)

            return h

        else:  # slave
            # 64 channel blocks:
            h = self.block1_1()
            h = self.block1_2(h)
    
            # 128 channel blocks:
            h = self.block2_1(h)
            h = self.block2_2(h)
    
            # 256 channel blocks:
            h = self.block3_1(h)
            h = self.block3_2(h)
            h = self.block3_3(h)
    
            # 512 channel blocks:
            h = self.block4_1(h)
            h = self.block4_2(h)
            h = self.block4_3(h)
    
            # 512 channel blocks:
            h = self.block5_1(h)
            h = self.block5_2(h)
            h = self.block5_3(h)

            return h
