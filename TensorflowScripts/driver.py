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

VFAE(rng=np.random.RandomState(), 
    input_source=theano.tensor.vector(),
    input_target=theano.tensor.vector(),
    label_source=theano.tensor.vector(),
    batch_size=200,
    struct=VFAE_struct(),
    coef=VFAE_coef(),
    train = True,
    init_params=None)
