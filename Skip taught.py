#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Skip-thought vectors
'''
import os

import theano
import theano.tensor as tensor

#import cPickle as pkl
import _pickle as cPickle
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

profile = False


def load_model(path_to_models, path_to_tables):
    """
    Load the model with saved tables
    """
    path_to_umodel = path_to_models + 'uni_skip.npz'
    path_to_bmodel = path_to_models + 'bi_skip.npz'

    # Load model options
    with open('%s.pkl'%path_to_umodel, 'rb') as f:
        uoptions = pkl.load(f)
    with open('%s.pkl'%path_to_bmodel, 'rb') as f:
        boptions = pkl.load(f)

    # Load parameters
    uparams = init_params(uoptions)
    uparams = load_params(path_to_umodel, uparams)
    utparams = init_tparams(uparams)
    bparams = init_params_bi(boptions)
    bparams = load_params(path_to_bmodel, bparams)
    btparams = init_tparams(bparams)

    # Extractor functions
    embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)
    f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
    embedding, x_mask, ctxw2v = build_encoder_bi(btparams, boptions)
    f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')

    # Tables
    utable, btable = load_tables(path_to_tables)

    # Store everything we need in a dictionary
    model = {}
    model['uoptions'] = uoptions
    model['boptions'] = boptions
    model['utable'] = utable
    model['btable'] = btable
    model['f_w2v'] = f_w2v
    model['f_w2v2'] = f_w2v2

    return model

def load_tables(path_to_tables):
    """
    Load the tables
    """
    words = []
    utable = numpy.load(path_to_tables + 'utable.npy')
    btable = numpy.load(path_to_tables + 'btable.npy')
    f = open(path_to_tables + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    btable = OrderedDict(zip(words, btable))
    return utable, btable

def encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
    """
    Encode sentences in the list X. Each entry will return a vector
    """
    # first, do preprocessing
    X = preprocess(X)

    # word dictionary and init
    d = defaultdict(lambda : 0)
    for w in model['utable'].keys():
        d[w] = 1
    ufeatures = numpy.zeros((len(X), model['uoptions']['dim']), dtype='float32')
    bfeatures = numpy.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i,s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print(k)
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]

            if use_eos:
                uembedding = numpy.zeros((k+1, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = numpy.zeros((k+1, len(caps), model['boptions']['dim_word']), dtype='float32')
            else:
                uembedding = numpy.zeros((k, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = numpy.zeros((k, len(caps), model['boptions']['dim_word']), dtype='float32')
            for ind, c in enumerate(caps):
                caption = captions[c]
                for j in range(len(caption)):
                    if d[caption[j]] > 0:
                        uembedding[j,ind] = model['utable'][caption[j]]
                        bembedding[j,ind] = model['btable'][caption[j]]
                    else:
                        uembedding[j,ind] = model['utable']['UNK']
                        bembedding[j,ind] = model['btable']['UNK']
                if use_eos:
                    uembedding[-1,ind] = model['utable']['<eos>']
                    bembedding[-1,ind] = model['btable']['<eos>']
            if use_eos:
                uff = model['f_w2v'](uembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))
            else:
                uff = model['f_w2v'](uembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))
            if use_norm:
                for j in range(len(uff)):
                    uff[j] /= norm(uff[j])
                    bff[j] /= norm(bff[j])
            for ind, c in enumerate(caps):
                ufeatures[c] = uff[ind]
                bfeatures[c] = bff[ind]
    
    features = numpy.c_[ufeatures, bfeatures]
    return features

def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X

def _p(pp, name):
    """
    make prefix-appended name
    """
    return '%s_%s'%(pp, name)

def init_tparams(params):
    """
    initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    """
    load parameters
    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        params[kk] = pp[kk]
    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'gru': ('param_init_gru', 'gru_layer')}

