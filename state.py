from collections import OrderedDict

def prototype_state():
    state = {}

    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    state['prefix'] = 'state'

    # These are unknown word placeholders
    state['oov'] = '<unk>'
    # Watch out for these
    state['unk_sym'] = 0
    state['eos_sym'] = 2
    state['sos_sym'] = 1

    state['n_samples'] = 40

    # These are end-of-sequence marks
    state['start_sym_sent'] = '<s>'
    state['end_sym_sent'] = '</s>'

    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'

    # ----- SIZES ----
    # Dimensionality of hidden layers
    state['qdim'] = 512
    # Dimensionality of low-rank approximation
    state['rankdim'] = 256

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003

    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'

    # Batch size
    state['bs'] = 128

    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 20

    # Maximum sequence length / trim batches
    state['seqlen'] = 50

    # Should we use a deep output layer
    # and maxout on the outputs?
    state['deep_out'] = True
    state['maxout_out'] = True

    state['step_type'] = 'gated'
    state['rec_activation'] = "lambda x: T.tanh(x)"

    # Maximum number of iterations
    state['max_iters'] = 10
    state['save_dir'] = './'

    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 10
    # Validation frequency
    state['validFreq'] = 5000
    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1
    return state

def prototype_test():
    state = prototype_state()

    state['train_sentences'] = "tests/data/test.word.train.pkl"
    state['valid_sentences'] = "tests/data/test.word.valid.pkl"
    state['dictionary'] = "tests/data/test.dict.pkl"
    state['save_dir'] = "tests/models/"

    state['prefix'] = "test_"

    state['deep_out'] = True
    state['maxout_out'] = False

    #
    state['qdim'] = 5
    # Dimensionality of low-rank approximation
    state['rankdim'] = 10

    state['bs'] = 10
    state['seqlen'] = 50
    return state

def prototype_train():
    state = {}
    state['train_sentences'] = "Data/train.word.pkl"
    state['valid_sentences'] = "Data/validation.word.pkl"
    state['dictionary'] = "dict.dict.pkl"
    state['save_dir'] = "model/"

    #state['initialize_from_pretrained_word_embeddings'] = True
    #state['pretrained_word_embeddings_file'] = './tests/data/MT_WordEmb.pkl'

    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    state['prefix'] = 'train_state_'

    state['patience'] = 2
    state['cost_threshold'] = 1.003

    state['rec_activation'] = "lambda x: T.tanh(x)"
    state['step_type'] = 'gated'

    state['rankdim'] = 128
    state['qdim'] = 256

    state['deep_out'] = True
    state['maxout_out'] = True

    state['unk_sym'] = 0
    state['eos_sym'] = 2
    state['sos_sym'] = 1

    state['cutoff'] = 1.

    state['updater'] = 'adam'
    state['bs'] = 128

    state['seqlen'] = 20

    state['loopIters'] = 10

    state['timeStop'] = 24*60

    state['sort_k_batches'] = 10
    state['trainFreq'] = 100
    state['validFreq'] = 3000

    return state


def prototype_largo():
    state = prototype_state()

    state['train_sentences'] = "Data/trn.word.pkl"
    state['valid_sentences'] = "Data/vld.word.pkl"
    state['dictionary'] = "Data/trn.dict.pkl"
    state['save_dir'] = "model/"
    state['prefix'] = "largo"

    return state

######################################################### BASE
def prototype_new():
    state = prototype_state()

    state['train_sentences'] = "Data/trn.word.pkl"
    state['valid_sentences'] = "Data/vld.word.pkl"
    state['dictionary'] = "Data/trn.dict.pkl"
    state['save_dir'] = "model/"
    state['prefix'] = "new_1_"
    state['n_samples'] = 10

    state['sort_k_batches'] = 10

    state['trainFreq'] = 10
    # Validation frequency
    state['validFreq'] = 50
    # Number of batches to process
    state['loopIters'] = 1000
    # Maximum number of minutes to run
    state['timeStop'] = 10

    return state

######################################################### UPDATER
def prototype_new_2():
    state = prototype_new()
    state['prefix'] = "new_2_"
    state['updater'] = 'rmsprop'
    return state

def prototype_new_3():
    state = prototype_new()
    state['prefix'] = "new_3_"
    state['updater'] = 'adadelta'
    return state

######################################################### ValidFreq
def prototype_new_4():
    state = prototype_new()
    state['prefix'] = "new_4_"
    state['validFreq'] = 100
    return state

def prototype_new_5():
    state = prototype_new()
    state['prefix'] = "new_5_"
    state['validFreq'] = 200
    return state

def prototype_new_6():
    state = prototype_new()
    state['prefix'] = "new_6_"
    state['validFreq'] = 500
    return state

######################################################### N CAPAS
def prototype_new_7():
    state = prototype_new()
    state['prefix'] = "new_7_"
    # Dimensionality of hidden layers
    state['qdim'] = 128
    # Dimensionality of low-rank approximation
    state['rankdim'] = 64
    return state

def prototype_new_8():
    state = prototype_new()
    state['prefix'] = "new_8_"
    # Dimensionality of hidden layers
    state['qdim'] = 256
    # Dimensionality of low-rank approximation
    state['rankdim'] = 128
    return state

def prototype_new_9():
    state = prototype_new()
    state['prefix'] = "new_9_"
    # Dimensionality of hidden layers
    state['qdim'] = 1024
    # Dimensionality of low-rank approximation
    state['rankdim'] = 512
    return state

def prototype_new_12():
    state = prototype_new()
    state['prefix'] = "new_12_"
    # Dimensionality of hidden layers
    state['qdim'] = 64
    # Dimensionality of low-rank approximation
    state['rankdim'] = 32
    return state

def prototype_new_13():
    state = prototype_new()
    state['prefix'] = "new_13_"
    # Dimensionality of hidden layers
    state['qdim'] = 2048
    # Dimensionality of low-rank approximation
    state['rankdim'] = 1024
    return state

######################################################### BS
def prototype_new_10():
    state = prototype_new_6()
    state['prefix'] = "new_10_"
    state['bs'] = 100
    return state

def prototype_new_11():
    state = prototype_new_6()
    state['prefix'] = "new_11_"
    state['bs'] = 150
    return state

def prototype_new_14():
    state = prototype_new_6()
    state['prefix'] = "new_14_"
    state['bs'] = 75
    return state

def prototype_new_15():
    state = prototype_new_6()
    state['prefix'] = "new_15_"
    state['bs'] = 60
    return state

def prototype_new_16():
    state = prototype_new_6()
    state['prefix'] = "new_16_"
    state['bs'] = 50
    return state

def prototype_new_17():
    state = prototype_new_6()
    state['prefix'] = "new_17_"
    state['bs'] = 25
    return state

######################################################### SORT K BATCHES 14
def prototype_new_18():
    state = prototype_new_14()
    state['prefix'] = "new_18_"
    state['sort_k_batches'] = 1
    return state

def prototype_new_19():
    state = prototype_new_14()
    state['prefix'] = "new_19_"
    state['sort_k_batches'] = 5
    return state

def prototype_new_20():
    state = prototype_new_14()
    state['prefix'] = "new_20_"
    state['sort_k_batches'] = 15
    return state

def prototype_new_21():
    state = prototype_new_14()
    state['prefix'] = "new_21_"
    state['sort_k_batches'] = 20
    return state

def prototype_new_22():
    state = prototype_new_14()
    state['prefix'] = "new_22_"
    state['sort_k_batches'] = 25
    return state

def prototype_new_25():
    state = prototype_new_14()
    state['prefix'] = "new_25_"
    state['sort_k_batches'] = 40
    return state

def prototype_new_23():
    state = prototype_new_14()
    state['prefix'] = "new_23_"
    state['sort_k_batches'] = 50
    return state

def prototype_new_24():
    state = prototype_new_14()
    state['prefix'] = "new_24_"
    state['sort_k_batches'] = 75
    return state

def prototype_new_25():
    state = prototype_new_14()
    state['prefix'] = "new_25_"
    state['sort_k_batches'] = 40
    return state

######################################################### SORT K BATCHES 10
def prototype_new_26():
    state = prototype_new_10()
    state['prefix'] = "new_26_"
    state['sort_k_batches'] = 1
    return state

def prototype_new_27():
    state = prototype_new_10()
    state['prefix'] = "new_27_"
    state['sort_k_batches'] = 25
    return state

######################################################### SORT K BATCHES 16
def prototype_new_28():
    state = prototype_new_16()
    state['prefix'] = "new_28_"
    state['sort_k_batches'] = 1
    return state

def prototype_new_29():
    state = prototype_new_16()
    state['prefix'] = "new_29_"
    state['sort_k_batches'] = 25
    return state

######################################################### FUNCION ACTIVACION
def prototype_new_30():
    state = prototype_new_10()
    state['prefix'] = "new_30_"
    state['rec_activation'] = "lambda x: T.nnet.sigmoid(x)"
    return state

def prototype_new_31():
    state = prototype_new_10()
    state['prefix'] = "new_31_"
    state['rec_activation'] = "lambda x: SoftMax(x)"
    return state

def prototype_new_36():
    state = prototype_new_14()
    state['prefix'] = "new_36_"
    state['rec_activation'] = "lambda x: T.nnet.sigmoid(x)"
    return state

def prototype_new_37():
    state = prototype_new_16()
    state['prefix'] = "new_37_"
    state['rec_activation'] = "lambda x: T.nnet.sigmoid(x)"
    return state



######################################################### DEEPOUT MAXOUT
def prototype_new_32():
    state = prototype_new_10()
    state['prefix'] = "new_32_"
    state['deep_out'] = True
    state['maxout_out'] = False
    return state

def prototype_new_33():
    state = prototype_new_10()
    state['prefix'] = "new_33_"
    state['deep_out'] = False
    state['maxout_out'] = False
    return state

def prototype_new_34():
    state = prototype_new_10()
    state['prefix'] = "new_34_"
    state['deep_out'] = False
    state['maxout_out'] = True
    return state

######################################################### SEQLEN
def prototype_new_35():
    state = prototype_new_37()
    state['prefix'] = "new_35_"
    state['seqlen'] = 5
    return state

def prototype_new_38():
    state = prototype_new_37()
    state['prefix'] = "new_38_"
    state['seqlen'] = 15
    return state

def prototype_new_39():
    state = prototype_new_37()
    state['prefix'] = "new_39_"
    state['seqlen'] = 20
    return state

def prototype_new_40():
    state = prototype_new_37()
    state['prefix'] = "new_40_"
    state['seqlen'] = 25
    return state

def prototype_new_41():
    state = prototype_new_37()
    state['prefix'] = "new_41_"
    state['seqlen'] = 75
    return state

def prototype_new_42():
    state = prototype_new_37()
    state['prefix'] = "new_42_"
    state['seqlen'] = 100
    return state

#########################################################
#########################################################
#########################################################
######################################################### MEDIANO
def prototype_mediano():
    state = prototype_new_39()
    state['prefix'] = "mediano_"
    # Number of batches to process
    state['loopIters'] = 50000
    # Maximum number of minutes to run
    state['timeStop'] = 300
    return state

######################################################### PATIENCE Y COST THRESHOLD
def prototype_mediano_2():
    state = prototype_mediano()
    state['prefix'] = "mediano_2_"
    state['patience'] = 1
    state['cost_threshold'] = 0.5
    return state

def prototype_mediano_3():
    state = prototype_mediano()
    state['prefix'] = "mediano_3_"
    state['patience'] = 2
    state['cost_threshold'] = 1.5
    return state

def prototype_mediano_4():
    state = prototype_mediano()
    state['prefix'] = "mediano_4_"
    state['patience'] = 3
    state['cost_threshold'] = 2.0
    return state

def prototype_mediano_5():
    state = prototype_mediano()
    state['prefix'] = "mediano_5_"
    state['patience'] = 4
    state['cost_threshold'] = 5.0
    return state
