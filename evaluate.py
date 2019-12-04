#!/usr/bin/env python
"""
Evaluation script.

For paper submissions, this script should normally be run with flags --exclude-sos --plot-graphs, and both with and without the flag --exclude-stop-words.
"""

import argparse
import cPickle
import traceback
import logging
import time
import sys
import signal

import os
import numpy
import codecs
import math

from recurrent_lm import *
from numpy_compat import argpartition
from state import *
from data_iterator import *

import matplotlib
matplotlib.use('Agg')
import pylab

logger = logging.getLogger(__name__)

# List of all 77 English pronouns, all puntucation signs included in MovieTriples and other special tokens.
stopwords = "all another any anybody anyone anything both each each other either everybody everyone everything few he her hers herself him himself his I it its itself many me mine more most much myself neither no one nobody none nothing one one another other others ours ourselves several she some somebody someone something that their theirs them themselves these they this those us we what whatever which whichever who whoever whom whomever whose you your yours yourself yourselves . , ? ' - -- ! <unk> </s> <s>"

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("test_path",
            type=str, help="File of test data")

    parser.add_argument("--exclude-sos", action="store_true",
                       help="Mask <s> from the cost computation")

    parser.add_argument("--plot-graphs", action="store_true",
                       help="Plots frequency graphs for word perplexity and pointwise mutual information")

    parser.add_argument("--exclude-stop-words", action="store_true",
                       help="Exclude stop words (English pronouns, puntucation signs and special tokens) from all metrics. These words make up approximate 48.37% of the training set, so removing them should focus the metrics on the topical content and ignore syntatic errors.")

    parser.add_argument("--document-ids",
                       type=str, help="File containing document ids for each triple (one id per line, if there are multiple tabs the first entry will be taken as the doc id). If this is given the script will compute standard deviations across documents for all metrics.")

    return parser.parse_args()

def load(model, filename):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename)
    signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)

def main():
    args = parse_args()
    state = prototype_state()
    #state = prototype_train()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    # This is a hack: we replace the validation set with the test set
    state['valid_triples'] = args.test_path
    state['valid_sentences'] = args.test_path

    rng = numpy.random.RandomState(state['seed'])
    model = RecurrentLM(rng, state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        load(model, model_path)
    else:
        raise Exception("Must specify a valid model path")

    eval_batch = model.build_eval_function()
    eval_misclass_batch = model.build_eval_misclassification_function()

    # Initialize list of stopwords to remove
    if args.exclude_stop_words:
        logger.debug("Initializing stop-word list")
        stopwords_lowercase = stopwords.lower().split(' ')
        stopwords_indices = []
        for word in stopwords_lowercase:
            if word in model.str_to_idx:
                stopwords_indices.append(model.str_to_idx[word])

    _, test_data = get_batch_iterator(rng, state)
    test_data.start()

    # Load document ids
    if args.document_ids:
        labels_file = open(args.document_ids, 'r')
        labels_text = labels_file.readlines()
        document_ids = numpy.zeros((len(labels_text)), dtype='int32')
        for i in range(len(labels_text)):
            document_ids[i] = int(labels_text[i].split('\t')[0])

        unique_document_ids = numpy.unique(document_ids)
        print 'test_data.data_len', test_data.data_len
        print 'document_ids', document_ids.shape
        assert(test_data.data_len == document_ids.shape[0])

    else:
        print 'Warning no file with document ids given... standard deviations cannot be computed.'
        document_ids = numpy.zeros((test_data.data_len), dtype='int32')
        unique_document_ids = numpy.unique(document_ids)

    # Variables to store test statistics
    test_cost = 0
    test_cost_first_utterances = 0
    test_cost_last_utterance_marginal = 0
    test_misclass = 0
    test_misclass_first_utterances = 0
    test_empirical_mutual_information = 0

    test_wordpreds_done = 0
    test_wordpreds_done_last_utterance = 0
    test_triples_done = 0

    # Prepare variables for plotting histogram over word-perplexities and mutual information
    test_data_len = test_data.data_len
    test_cost_list = numpy.zeros((test_data_len,))
    test_pmi_list = numpy.zeros((test_data_len,))

    test_cost_last_utterance_marginal_list = numpy.zeros((test_data_len,))
    test_misclass_list = numpy.zeros((test_data_len,))
    test_misclass_last_utterance_list = numpy.zeros((test_data_len,))

    words_in_triples_list = numpy.zeros((test_data_len,))
    words_in_last_utterance_list = numpy.zeros((test_data_len,))

    # Prepare variables for printing the test examples the model performs best and worst on
    test_extrema_setsize = 100
    test_extrema_samples_to_print = 20

    test_lowest_costs = numpy.ones((test_extrema_setsize,))*1000
    test_lowest_triples = numpy.ones((test_extrema_setsize,state['seqlen']))*1000
    test_highest_costs = numpy.ones((test_extrema_setsize,))*(-1000)
    test_highest_triples = numpy.ones((test_extrema_setsize,state['seqlen']))*(-1000)

    logger.debug("[TEST START]")

    while True:
        batch = test_data.next()
        # Train finished
        if not batch:
            break

        logger.debug("[TEST] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))

        x_data = batch['x']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']

        # Hack to get rid of start of sentence token.
        if args.exclude_sos and model.sos_sym != -1:
            x_cost_mask[x_data == model.sos_sym] = 0

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask[x_data == word_index] = 0

        batch['num_preds'] = numpy.sum(x_cost_mask)

        c, c_list = eval_batch(x_data, max_length, x_cost_mask)

        c_list = c_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        c_list = numpy.sum(c_list, axis=1)

        non_nan_entries = numpy.array(c_list >= 0, dtype=int)
        c_list[numpy.where(non_nan_entries==0)] = 0

        #words_in_triples = numpy.sum(x_cost_mask, axis=0)
        #c_list = c_list / words_in_triples


        if numpy.isinf(c) or numpy.isnan(c):
            continue

        test_cost += c

        # Store test costs in list
        nxt =  min((test_triples_done+batch['x'].shape[1]), test_data_len)
        triples_in_batch = nxt-test_triples_done

        words_in_triples = numpy.sum(x_cost_mask, axis=0)
        words_in_triples_list[(nxt-triples_in_batch):nxt] = words_in_triples[0:triples_in_batch]

        #print 'words_in_triples', words_in_triples.shape, words_in_triples
        # We don't need to normalzie by the number of words... not if we're computing standard deviations at least...
        #c_list = c_list / words_in_triples

        #test_cost_list[(nxt-triples_in_batch):nxt] = numpy.exp(c_list[0:triples_in_batch])
        test_cost_list[(nxt-triples_in_batch):nxt] = c_list[0:triples_in_batch]

        # Store best and worst test costs
        con_costs = numpy.concatenate([test_lowest_costs, c_list[0:triples_in_batch]])
        con_triples = numpy.concatenate([test_lowest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
        con_indices = con_costs.argsort()[0:test_extrema_setsize][::1]
        test_lowest_costs = con_costs[con_indices]
        test_lowest_triples = con_triples[con_indices]

        con_costs = numpy.concatenate([test_highest_costs, c_list[0:triples_in_batch]])
        con_triples = numpy.concatenate([test_highest_triples, x_data[:, 0:triples_in_batch].T], axis=0)
        con_indices = con_costs.argsort()[-test_extrema_setsize:][::-1]
        test_highest_costs = con_costs[con_indices]
        test_highest_triples = con_triples[con_indices]

        # Compute word-error rate
        miscl, miscl_list = eval_misclass_batch(x_data, max_length, x_cost_mask)
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        test_misclass += miscl

        # Store misclassification errors in list
        miscl_list = miscl_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        miscl_list = numpy.sum(miscl_list, axis=1)
        test_misclass_list[(nxt-triples_in_batch):nxt] = miscl_list[0:triples_in_batch]

        # Equations to compute empirical mutual information

        # Compute marginal log-likelihood of last utterance in triple:
        # We approximate it with the margina log-probabiltiy of the utterance being observed first in the triple
        x_data_last_utterance = batch['x_last_utterance']
        x_cost_mask_last_utterance = batch['x_mask_last_utterance']
        x_start_of_last_utterance = batch['x_start_of_last_utterance']

        # Hack to get rid of start of sentence token.
        if args.exclude_sos and model.sos_sym != -1:
            x_cost_mask_last_utterance[x_data_last_utterance == model.sos_sym] = 0

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask_last_utterance[x_data_last_utterance == word_index] = 0


        words_in_last_utterance = numpy.sum(x_cost_mask_last_utterance, axis=0)
        words_in_last_utterance_list[(nxt-triples_in_batch):nxt] = words_in_last_utterance[0:triples_in_batch]

        batch['num_preds_at_utterance'] = numpy.sum(x_cost_mask_last_utterance)

        marginal_last_utterance_loglikelihood, marginal_last_utterance_loglikelihood_list = eval_batch(x_data_last_utterance, max_length, x_cost_mask_last_utterance)

        marginal_last_utterance_loglikelihood_list = marginal_last_utterance_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_last_utterance_loglikelihood_list = numpy.sum(marginal_last_utterance_loglikelihood_list, axis=1)
        test_cost_last_utterance_marginal_list[(nxt-triples_in_batch):nxt] = marginal_last_utterance_loglikelihood_list[0:triples_in_batch]

        # Compute marginal log-likelihood of first utterances in triple by masking the last utterance
        x_cost_mask_first_utterances = numpy.copy(x_cost_mask)
        for i in range(batch['x'].shape[1]):
            x_cost_mask_first_utterances[x_start_of_last_utterance[i]:max_length, i] = 0

        marginal_first_utterances_loglikelihood, marginal_first_utterances_loglikelihood_list = eval_batch(x_data, max_length, x_cost_mask_first_utterances)

        marginal_first_utterances_loglikelihood_list = marginal_first_utterances_loglikelihood_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        marginal_first_utterances_loglikelihood_list = numpy.sum(marginal_first_utterances_loglikelihood_list, axis=1)

        # Compute empirical mutual information and pointwise empirical mutual information
        test_empirical_mutual_information += -c + marginal_first_utterances_loglikelihood + marginal_last_utterance_loglikelihood
        test_pmi_list[(nxt-triples_in_batch):nxt] = (-c_list*words_in_triples + marginal_first_utterances_loglikelihood_list + marginal_last_utterance_loglikelihood_list)[0:triples_in_batch]

        # Store log P(U_1, U_2) cost computed during mutual information
        test_cost_first_utterances += marginal_first_utterances_loglikelihood

        # Store marginal log P(U_3)
        test_cost_last_utterance_marginal += marginal_last_utterance_loglikelihood


        # Compute word-error rate for first utterances
        miscl_first_utterances, miscl_first_utterances_list = eval_misclass_batch(x_data, max_length, x_cost_mask_first_utterances)
        test_misclass_first_utterances += miscl_first_utterances
        if numpy.isinf(c) or numpy.isnan(c):
            continue

        # Store misclassification for last utterance
        miscl_first_utterances_list = miscl_first_utterances_list.reshape((batch['x'].shape[1],max_length), order=(1,0))
        miscl_first_utterances_list = numpy.sum(miscl_first_utterances_list, axis=1)

        miscl_last_utterance_list = miscl_list - miscl_first_utterances_list

        test_misclass_last_utterance_list[(nxt-triples_in_batch):nxt] = miscl_last_utterance_list[0:triples_in_batch]

        test_wordpreds_done += batch['num_preds']
        test_wordpreds_done_last_utterance += batch['num_preds_at_utterance']
        test_triples_done += batch['num_triples']

    logger.debug("[TEST END]")

    test_cost_last_utterance_marginal /= test_wordpreds_done_last_utterance
    test_cost_last_utterance = (test_cost - test_cost_first_utterances) / test_wordpreds_done_last_utterance
    test_cost /= test_wordpreds_done
    test_cost_first_utterances /= float(test_wordpreds_done - test_wordpreds_done_last_utterance)

    test_misclass_last_utterance = float(test_misclass - test_misclass_first_utterances) / float(test_wordpreds_done_last_utterance)
    test_misclass_first_utterances /= float(test_wordpreds_done - test_wordpreds_done_last_utterance)
    test_misclass /= float(test_wordpreds_done)
    test_empirical_mutual_information /= float(test_triples_done)

    print "** test cost (NLL) = %.4f, test word-perplexity = %.4f, test word-perplexity last utterance = %.4f, test word-perplexity marginal last utterance = %.4f, test mean word-error = %.4f, test mean word-error last utterance = %.4f, test emp. mutual information = %.4f" % (float(test_cost), float(math.exp(test_cost)), float(math.exp(test_cost_last_utterance)), float(math.exp(test_cost_last_utterance_marginal)),float(test_misclass), float(test_misclass_last_utterance), test_empirical_mutual_information)

    # Plot histogram over test costs
    if args.plot_graphs:
        try:
            pylab.figure()
            bins = range(0, 50, 1)
            pylab.hist(numpy.exp(test_cost_list), normed=1, histtype='bar')
            pylab.savefig(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + 'Test_WordPerplexities.png')
        except:
            pass

    # Print 5 of 10% test samples with highest log-likelihood
    # TODO: There is a problem in printing the words. The extra white spacing should be removed...
    if args.plot_graphs:
        print " highest word log-likelihood test samples: "
        numpy.random.shuffle(test_lowest_triples)
        for i in range(test_extrema_samples_to_print):
            print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(test_lowest_triples[i,:]))))

        print " lowest word log-likelihood test samples: "
        numpy.random.shuffle(test_highest_triples)
        for i in range(test_extrema_samples_to_print):
            print "      Sample: {}".format(" ".join(model.indices_to_words(numpy.ravel(test_highest_triples[i,:]))))


    # Plot histogram over empirical pointwise mutual informations
    if args.plot_graphs:
        try:
            pylab.figure()
            bins = range(0, 100, 1)
            pylab.hist(test_pmi_list, normed=1, histtype='bar')
            pylab.savefig("tests" + '/' + 'Test_PMI.png')
        except:
            pass


    #print 'words_in_triples_list', words_in_triples_list.shape
    #print 'words_in_last_utterance_list', words_in_last_utterance_list.shape
    #print 'test_cost_list', test_cost_list.shape
    #print 'test_cost_last_utterance_marginal_list', test_cost_last_utterance_marginal_list.shape
    #print 'test_misclass_list', test_misclass_list.shape
    #print 'test_misclass_last_utterance_list', test_misclass_last_utterance_list.shape

    per_document_test_cost = numpy.zeros((len(unique_document_ids)), dtype='float32')
    per_document_test_cost_last_utterance = numpy.zeros((len(unique_document_ids)), dtype='float32')

    per_document_test_misclass = numpy.zeros((len(unique_document_ids)), dtype='float32')
    per_document_test_misclass_last_utterance = numpy.zeros((len(unique_document_ids)), dtype='float32')

    all_words_squared = 0
    all_words_in_last_utterance_squared = 0
    for doc_id in range(len(unique_document_ids)):
        doc_indices = numpy.where(document_ids == unique_document_ids[doc_id])

        per_document_test_cost[doc_id] = numpy.sum(test_cost_list[doc_indices]) / numpy.sum(words_in_triples_list[doc_indices])
        per_document_test_cost_last_utterance[doc_id] = numpy.sum(test_cost_last_utterance_marginal_list[doc_indices]) / numpy.sum(words_in_last_utterance_list[doc_indices])

        per_document_test_misclass[doc_id] = numpy.sum(test_misclass_list[doc_indices]) / numpy.sum(words_in_triples_list[doc_indices])
        per_document_test_misclass_last_utterance[doc_id] = numpy.sum(test_misclass_last_utterance_list[doc_indices]) / numpy.sum(words_in_last_utterance_list[doc_indices])

        all_words_squared += float(numpy.sum(words_in_triples_list[doc_indices]))**2
        all_words_in_last_utterance_squared += float(numpy.sum(words_in_last_utterance_list[doc_indices]))**2

    assert(numpy.sum(words_in_triples_list) == test_wordpreds_done)
    assert(numpy.sum(words_in_last_utterance_list) == test_wordpreds_done_last_utterance)

    print 'per_document_test_cost', per_document_test_cost
    print 'per_document_test_misclass', per_document_test_misclass
    print 'all_words_squared', all_words_squared
    print 'all_words_in_last_utterance_squared', all_words_in_last_utterance_squared
    print 'test_wordpreds_done', test_wordpreds_done


    per_document_test_cost_variance = numpy.var(per_document_test_cost) * float(all_words_squared) / float(test_wordpreds_done**2)
    per_document_test_cost_last_utterance_variance = numpy.var(per_document_test_cost_last_utterance) * float(all_words_in_last_utterance_squared) / float(test_wordpreds_done_last_utterance**2)
    per_document_test_misclass_variance = numpy.var(per_document_test_misclass) * float(all_words_squared) / float(test_wordpreds_done**2)
    per_document_test_misclass_last_utterance_variance = numpy.var(per_document_test_misclass_last_utterance) * float(all_words_in_last_utterance_squared) / float(test_wordpreds_done_last_utterance**2)

    print 'Standard deviations:'
    print "** test cost (NLL) = ", math.sqrt(per_document_test_cost_variance)
    print "** test perplexity (NLL) = ", math.sqrt((math.exp(per_document_test_cost_variance) - 1)*math.exp(2*test_cost+per_document_test_cost_variance))

    print "** test cost last utterance (NLL) = ", math.sqrt(per_document_test_cost_last_utterance_variance)
    print "** test perplexity last utterance  (NLL) = ", math.sqrt((math.exp(per_document_test_cost_last_utterance_variance) - 1)*math.exp(2*test_cost+per_document_test_cost_last_utterance_variance))

    print "** test word-error = ", math.sqrt(per_document_test_misclass_variance)
    print "** test last utterance word-error = ", math.sqrt(per_document_test_misclass_last_utterance_variance)

    logger.debug("All done, exiting...")

if __name__ == "__main__":
    main()
