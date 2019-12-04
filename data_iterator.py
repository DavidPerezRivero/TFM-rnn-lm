import numpy as np
import theano
import theano.tensor as T
import sys, getopt
import logging

from state import *
from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime

logger = logging.getLogger(__name__)

def create_padded_batch(state, x):
    mx = state['seqlen']
    n = state['bs']

    X = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32')

    # Variables to store last utterance (for computing mutual information metric)
    X_last_utterance = numpy.zeros((mx, n), dtype='int32')
    Xmask_last_utterance = numpy.zeros((mx, n), dtype='float32')
    X_start_of_last_utterance = numpy.zeros((n), dtype='int32')

    # Fill X and Xmask
    # Keep track of number of predictions and maximum triple length
    num_preds = 0
    num_preds_last_utterance = 0
    max_length = 0
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        sent_length = len(x[0][idx])

        if mx < sent_length:
            continue

        X[:sent_length, idx] = x[0][idx][:sent_length]

        max_length = max(max_length, sent_length)
        # Set the number of predictions == sum(Xmask), for cost purposes
        num_preds += sent_length

        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            X[sent_length:, idx] = state['eos_sym']

        # Initialize Xmask column with ones in all positions that
        # were just set in X
        Xmask[:sent_length, idx] = 1.

        sos_indices = numpy.where(X[:, idx] == state['sos_sym'])[0]
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]

        # Find start of last utterance and store the utterance
        assert (len(eos_indices) >= len(sos_indices))

        if len(sos_indices) > 0: # Check that dialogue is not empty
            start_of_last_utterance = sos_indices[-1]
        else: # If it is empty, then we define last utterance to start at the beginning
            start_of_last_utterance = 0

        num_preds_last_utterance += sent_length - start_of_last_utterance

        X_start_of_last_utterance[idx] = start_of_last_utterance
        X_last_utterance[0:(sent_length-start_of_last_utterance), idx] = X[start_of_last_utterance:sent_length, idx]
        Xmask_last_utterance[0:(sent_length-start_of_last_utterance), idx] = Xmask[start_of_last_utterance:sent_length, idx]


    assert num_preds == numpy.sum(Xmask)
    return {'x': X,                                                 \
            'x_mask': Xmask,                                        \
            'num_preds': num_preds,                                 \
            'x_last_utterance': X_last_utterance,                   \
            'x_mask_last_utterance': Xmask_last_utterance,          \
            'x_start_of_last_utterance': X_start_of_last_utterance, \
            'num_preds_ast_utterance': num_preds_last_utterance,    \
            'num_triples': len(x[0]),                               \
            'max_length': max_length}

def get_batch_iterator(rng, state):
    class Iterator(SSIterator):
        def __init__(self, *args, **kwargs):
            SSIterator.__init__(self, rng, *args, **kwargs)
            self.batch_iter = None

        def get_homogenous_batch_iter(self):
            while True:
                k_batches = state['sort_k_batches']
                batch_size = state['bs']

                data = []
                for k in range(k_batches):
                    batch = SSIterator.next(self)
                    if batch:
                        data.append(batch)

                if not len(data):
                    return

                triples = data
                x = numpy.asarray(list(itertools.chain(*triples)))
                lens = numpy.asarray([map(len, x)])
                order = numpy.argsort(lens.max(axis=0)) if state['sort_k_batches'] > 1 \
                        else numpy.arange(len(x))

                for k in range(len(triples)):
                    indices = order[k * batch_size:(k + 1) * batch_size]
                    batch = create_padded_batch(state, [x[indices]])
                    if batch:
                        yield batch

        def start(self):
            SSIterator.start(self)
            self.batch_iter = None

        def next(self):
            if not self.batch_iter:
                self.batch_iter = self.get_homogenous_batch_iter()
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                return None
            return batch

    train_data = Iterator(
        batch_size=int(state['bs']),
        triple_file=state['train_sentences'],
        queue_size=100,
        use_infinite_loop=True,
        max_len=state['seqlen'])

    valid_data = Iterator(
        batch_size=int(state['bs']),
        triple_file=state['valid_sentences'],
        use_infinite_loop=False,
        queue_size=100,
        max_len=state['seqlen'])

    return train_data, valid_data
