from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from neural_nets.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from neural_nets.training import create_basic_model, create_cs_model
from neural_nets.beam_search_adapter import BeamSearchAdapter
from neural_nets.cp_model import GreedyAdapter
from argparse import ArgumentParser
import os
import time
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def _create_parser():
    parser = ArgumentParser(description="Load a model and use it for generation of rotowire summaries.")
    parser.add_argument( '--path'
                       , type=str
                       , required=True
                       , help="Path to prepared dataset files")
    parser.add_argument( '--output_path'
                       , help="path where to save generated summaries as well as the gold ones" 
                       , type=str
                       , required=True)
    parser.add_argument( '--batch_size'
                       , type=int
                       , default=16)
    parser.add_argument( '--checkpoint_path'
                       , help="path to directory with model checkpoints"
                       , type=str
                       , required=True)
    parser.add_argument( '--word_emb_dim'
                       , help="embedding dimensionality to which to project the words from the summary " +\
                              "and value and entity part of the input records"
                       , type=int
                       , default=128)
    parser.add_argument( '--tp_emb_dim'
                       , help="embedding dimensionality to which to project the type part of " +\
                              "input record"
                       , type=int
                       , default=128)
    parser.add_argument( '--ha_emb_dim'
                       , help="embedding dimensionality to which project the home/away flag part of " +\
                              "input record"
                       , type=int
                       , default=128)
    parser.add_argument( '--hidden_size'
                       , help="dimensionality of the hidden states"
                       , type=int
                       , default=300)
    parser.add_argument( '--attention_type'
                       , help="one of [dot, concat], specifies whether to use Luong style dot attention " +\
                              "or Bahdanau style concat attention"
                       , type=str
                       , default="dot")
    parser.add_argument( '--decoder_type'
                       , help="one of [joint, baseline], specifies whether to use baseline decoder " +\
                              "or decoder with joint-copy mechanism"
                       , type=str
                       , default="joint")
    parser.add_argument( '--dropout_rate'
                       , type=float
                       , help="rate at which to drop cells and corresponding connections at " +\
                              "the outputs of the internal LSTM layers of the model (number " +\
                              "between 0 and 1, 0 means no dropout)"
                       , default=0.2)
    parser.add_argument( '--with_cp'
                       , help="use EncoderDecoderContentSelection, with content planning"
                       , action='store_true')
    parser.add_argument( '--with_cs'
                       , help="create EncoderDecoderBaseline with Content Selection Encoder (CopyCS model)"
                       , action='store_true')
    parser.add_argument( '--with_csbidir'
                       , help="create EncoderDecoderBaseline with Content Selection Encoder and " +\
                              "bidirectional LSTM over it (CopyCSBidir model)"
                       , action='store_true')
    parser.add_argument( '--beam_search'
                       , help="use beam search decoding"
                       , action='store_true')
    parser.add_argument( '--beam_size'
                       , help="number of hypotheses to keep during the beam search"
                       , type=int
                       , default=5)
    parser.add_argument( '--from_gold_cp'
                       , help="generate from gold content plans instead of from generated ones" 
                       , action='store_true')
    return parser

def add_dummy_content_plans( dataset
                           , preprocess_cp_size):
    """Add tf.zeros as content plans to each batch of the dataset
    
    Since the test dataset doesn't contain gold content plans and we want to 
    keep symettry, we append tf.zeros as dummy content plans
    
    Args:
        dataset:            dataset to which we append the content plans
        preprocess_cp_size: the shape[1] of the appended content plans
    """
    def map_fn(summaries, types, entities, values, has):
        return summaries, tf.zeros(shape=(summaries.shape[0], preprocess_cp_size), dtype=tf.int16),\
               types, entities, values, has
    dataset = dataset.map(map_fn)
    return dataset


def beam_search( model
               , dataset
               , beam_size
               , eos
               , max_cp_size):
    """ Construct BeamSearchAdapter from the model let it predict hypotheses and decode the best
        hypothesis
    
    Args:
        model:          trained EncoderDecoderBasic or EncoderDecoderContentSelection
        dataset:        the dataset on which we want to predict
        beam_size:      number of hypotheses to keep during the beam search
        eos:            vocabulary index of the end of sequence token
        max_cp_size:    maximal length of a content plan
    """
    model = BeamSearchAdapter( model
                             , beam_size
                             , eos
                             , max_cp_size)
    model.compile()
    predictions = None
    batch_ix = 1
    start = time.time()
    for batch in dataset:
        # beam search decoding
        out_sentence, out_predecessors = model(batch)
        actual_predictions = np.zeros(shape=out_sentence[:, :, 0].shape, dtype=np.int16)

        # decode the out_sentence by backtracking from the last state
        for batch_dim in range(out_sentence.shape[0]):
            best = 0
            sum = out_sentence[batch_dim]
            pred = out_predecessors[batch_dim]
            for index in range(out_sentence.shape[1] - 1, -1, -1):
                actual_predictions[batch_dim, index] = sum[index, best].numpy()
                best = pred[index, best].numpy()
        # append to predictions
        if predictions is None:
            predictions = actual_predictions
        else:
            predictions = np.append(predictions, actual_predictions, axis=0)
        
        if (batch_ix % 10) == 0:
            end = time.time()
            print(f"batch {batch_ix} generated, elapsed {end-start} seconds")
            start = end
        batch_ix += 1
    
    return predictions


def generate( model
            , dataset
            , file_prefix
            , dir_path
            , eos
            , ix_to_tk
            , use_beam_search : bool = False
            , beam_size : int = 5
            , max_cp_size : int = 80
            , csap_model : bool = False
            , from_gold_cp : bool = False
            , add_cp : bool = False):
    """ Generate the summaries of the tables in the dataset

    Args:
        model:              trained EncoderDecoderBasic or EncoderDecoderContentSelection
        dataset:            the dataset on which we want to predict
        file_prefix:        the prefix of a filename where we save the generated sequences
        dir_path:           the directory where to save ${file_prefix}_golds.txt and ${file_prefix}_preds.txt
        eos:                vocabulary index of the end of sequence token
        ix_to_tk:           vocabulary used to decode words from indices
        use_beam_search:    whether to use beam search or not
        beam_size:          number of hypotheses to keep during the beam search
        max_cp_size:        maximal length of a content plan
        csap_model:         whether the trained model uses content planning mechanism or not
        from_gold_cp:       generate from gold content plans instead from the generated ones
        add_cp:             append dummy content plans to each batch in the dataset

    """

    # predict
    if csap_model and not use_beam_search:
        model = GreedyAdapter(model, max_cp_size, from_gold_cp)
        model.compile()
    if add_cp:
        dataset = add_dummy_content_plans(dataset, max_cp_size)
    if not use_beam_search:
        predictions = model.predict(dataset)
    else:
        predictions = beam_search( model
                                 , dataset
                                 , beam_size
                                 , eos
                                 , max_cp_size)
    # create targets
    targets = None
    for tgt in dataset.as_numpy_iterator():
        summaries, *_ = tgt
        if targets is None:
            targets = summaries[:, 1:]
        else:
            targets = np.append(targets, summaries[:, 1:], axis=0)
    
    # transform sequences of indices to words
    predictions_for_bleu = []
    targets_for_bleu = []
    for rp, tgt in zip(predictions, targets):
        # find the first occurrence of <<EOS>> token
        tpl = np.nonzero(tgt == eos)[0]
        ix = tpl[0] if len(tpl) > 0 else tgt.shape[0]
        targets_for_bleu.append([[ix_to_tk[i] for i in tgt[:ix]]])
        tpl = np.nonzero(rp == eos)[0]
        ix = tpl[0] if len(tpl) > 0 else ix
        predictions_for_bleu.append([ix_to_tk[i] for i in rp[:ix]])

    # save the generated summaries and gold summaries
    with open(os.path.join(dir_path, file_prefix + "preds.txt"), 'w', encoding='utf8') as f:
        for pred in predictions_for_bleu:
            print(" ".join(pred), file=f)

    with open(os.path.join(dir_path, file_prefix +"golds.txt"), 'w', encoding='utf8') as f:
        for tgt in targets_for_bleu:
            print(" ".join(tgt[0]), file=f)

    # calculate bleu
    bleu = corpus_bleu( targets_for_bleu
                      , predictions_for_bleu
                      , smoothing_function=SmoothingFunction().method4)
    print(f"BLEU : {bleu}")

def _main(args):
    config_path = os.path.join(args.path, "config.txt")
    valid_path = os.path.join(args.path, "valid.tfrecord")
    test_path = os.path.join(args.path, "test.tfrecord")
    vocab_path = os.path.join(args.path, "all_vocab.txt")
    for key, value in vars(args).items():
        print(f"{key} : {value}")

    # load config file
    max_table_size, max_summary_size, max_cp_size = \
            load_values_from_config( config_path, load_cp=args.with_cp)
    batch_size = args.batch_size

    # load validation dataset
    val_dataset, val_steps, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=valid_path
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=False
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size
                                    , preprocess_cp_size=max_cp_size
                                    , with_content_plans=args.with_cp)
    
    # test dataset is always WITHOUT content plans - as the network should
    # learn to generate the content plans and not to leverage the obtained ones
    test_dataset, test_steps, *dummies = load_tf_record_dataset( test_path
                                                               , vocab_path
                                                               , batch_size
                                                               , False # shuffle
                                                               , max_table_size
                                                               , max_summary_size)
    _ = (val_steps, pad, bos, test_dataset, test_steps, dummies)
    word_vocab_size = len(tk_to_ix)
    word_emb_dim = args.word_emb_dim
    tp_vocab_size = len(tp_to_ix)
    tp_emb_dim = args.tp_emb_dim
    ha_vocab_size = len(ha_to_ix)
    ha_emb_dim = args.ha_emb_dim
    entity_span = 22
    hidden_size = args.hidden_size

    checkpoint_dir = args.checkpoint_path

    # prepare attention callable
    if args.attention_type=="concat":
        attention = lambda: ConcatAttention(hidden_size)
    elif args.attention_type=="dot":
        attention = DotAttention
    else:
        attention = DotAttention

    # prepare decoder callable
    if args.decoder_type=="baseline":
        decoder_rnn = DecoderRNNCell
    elif args.decoder_type=="joint":
        print("\n\nChoosing RNNCellJointCopy\n\n")
        decoder_rnn = DecoderRNNCellJointCopy
    else:
        decoder_rnn = DecoderRNNCell
    
    # create and compile models
    if not args.with_cp:
        model = create_basic_model( args.batch_size
                                  , word_emb_dim
                                  , word_vocab_size
                                  , tp_emb_dim
                                  , tp_vocab_size
                                  , ha_emb_dim
                                  , ha_vocab_size
                                  , entity_span
                                  , hidden_size
                                  , attention
                                  , decoder_rnn
                                  , args.dropout_rate
                                  , encoder_cs_flag=args.with_cs
                                  , encoder_cs_bidir_flag=args.with_csbidir
                                  , max_table_size=max_table_size)
        # compile the model - enables eager execution (my custom change)
        model.compile( tf.keras.optimizers.Adam()
                     , tf.keras.losses.SparseCategoricalCrossentropy()
                     , 1.0, 100, 50) # just some dummy values
    else:
        model = create_cs_model( args.batch_size
                               , max_table_size
                               , word_emb_dim
                               , word_vocab_size
                               , tp_emb_dim
                               , tp_vocab_size
                               , ha_emb_dim
                               , ha_vocab_size
                               , hidden_size
                               , attention
                               , decoder_rnn
                               , args.dropout_rate)
        # compile the model - enables eager execution (my custom change)
        model.compile( tf.keras.optimizers.Adam()
                     , tf.keras.optimizers.Adam()
                     , tf.keras.losses.SparseCategoricalCrossentropy()
                     , tf.keras.losses.SparseCategoricalCrossentropy()
                     , 1.0, 100, 50) # just some dummy values
    
    # load the model from checkpoint and fail if the checkpoint is not resolved
    print(f"loading checkpoint from {checkpoint_dir}")
    print(f"latest checkpoint in the dir is {tf.train.latest_checkpoint(checkpoint_dir)}")
    checkpoint = tf.train.Checkpoint( model=model )
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print(status.assert_existing_objects_matched())

    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    pairs = [(val_dataset, "val_")]
    # if generating from gold content plans we can use only the validation dataset
    if not args.from_gold_cp:
      pairs.append((test_dataset, "test_"))

    # generate summaries and calculate bleu
    for data, file_prefix in pairs:
        generate( model
                , data
                , file_prefix
                , args.output_path
                , eos
                , ix_to_tk
                , use_beam_search=args.beam_search
                , beam_size=args.beam_size
                , max_cp_size=max_cp_size
                , csap_model=args.with_cp
                , from_gold_cp=args.from_gold_cp
                , add_cp=(args.with_cp and (file_prefix == "test_")))
    

if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
