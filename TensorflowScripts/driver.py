from __future__ import print_function

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

import nnet as nn
import criteria as er
import util

import VFAE_struct
import VFAE_coef
import VFAE_params
import VFAE

import pandas as pd

file_path ='Data Sets/CelebA/annot/list_attr_celeba.txt'
images_path = 'Data Sets/CelebA/img_align_celeba/'
cele_attrib = pd.read_csv(file_path,delimiter = "\s+",names = columns)

VFAE(rng=numpy.random.RandomState(), 
    input_source,
    input_target,
    label_source,
    batch_size=200,
    struct=VFAE_struct(),
    coef=VFAE_coef(),
    train = True,
    init_params=None)


"""Initialize the parameters for the multilayer perceptron
           :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input_source: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
        
        :type input_target: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)        
        :type xxx_struct: class NN_struct
        :param xxx_strucat: define the structure of each NN
        """

'''
VFAE_training(source_data, target_data, n_train_batches, n_epochs, struct, coef, description, process_display=True)