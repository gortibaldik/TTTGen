from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from neural_nets.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from neural_nets.training import create_basic_model, create_cs_model
from neural_nets.baseline_model import BeamSearchAdapter
from neural_nets.cp_model import GreedyAdapter
from argparse import ArgumentParser
import os
import time
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def _create_parser():
    parser = ArgumentParser()
    parser.add_argument( '--path'
                       , type=str
                       , required=True
                       , help="Path to tfrecord datasets")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--word_emb_dim', type=int, default=128)
    parser.add_argument('--tp_emb_dim', type=int, default=128)
    parser.add_argument('--ha_emb_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--attention_type', type=str, default="dot")
    parser.add_argument('--decoder_type', type=str, default="joint")
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--with_cp', action='store_true')
    parser.add_argument('--with_cs', action='store_true')
    parser.add_argument('--with_csbidir', action='store_true')
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5)
    return parser

def add_dummy_content_plans( dataset
                           , preprocess_cp_size):
    def map_fn(summaries, types, entities, values, has):
        return summaries, tf.zeros(shape=(summaries.shape[0], preprocess_cp_size), dtype=tf.int16),\
               types, entities, values, has
    dataset = dataset.map(map_fn)
    return dataset


def beam_search( model
               , dataset
               , beam_size
               , eos):
    model = BeamSearchAdapter( model
                             , beam_size
                             , eos)
    model.compile()
    predictions = None
    batch_ix = 1
    start = time.time()
    for batch in dataset:
        out_sentence, out_predecessors = model(batch)
        actual_predictions = np.zeros(shape=out_sentence[:, :, 0].shape, dtype=np.int16)
        for batch_dim in range(out_sentence.shape[0]):
            best = 0
            sum = out_sentence[batch_dim]
            pred = out_predecessors[batch_dim]
            for index in range(out_sentence.shape[1] - 1, -1, -1):
                actual_predictions[batch_dim, index] = sum[index, best].numpy()
                best = pred[index, best].numpy()
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
            , max_cp_size = None
            , csap_model : bool = False):
    if csap_model:
        model = GreedyAdapter(model)
        model.compile()
    if max_cp_size is not None:
        dataset = add_dummy_content_plans(dataset, max_cp_size)
    if not use_beam_search:
        predictions = model.predict(dataset)
    else:
        predictions = beam_search( model
                                 , dataset
                                 , beam_size
                                 , eos)
    targets = None
    for tgt in dataset.as_numpy_iterator():
        summaries, *_ = tgt
        if targets is None:
            targets = summaries[:, 1:]
        else:
            targets = np.append(targets, summaries[:, 1:], axis=0)
    
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

    with open(os.path.join(dir_path, file_prefix + "preds.txt"), 'w', encoding='utf8') as f:
        for pred in predictions_for_bleu:
            print(" ".join(pred), file=f)

    with open(os.path.join(dir_path, file_prefix +"golds.txt"), 'w', encoding='utf8') as f:
        for tgt in targets_for_bleu:
            print(" ".join(tgt[0]), file=f)

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
    max_table_size, max_summary_size, max_cp_size = \
            load_values_from_config( config_path, load_cp=args.with_cp)
    batch_size = args.batch_size
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

    if args.attention_type=="concat":
        attention = lambda: ConcatAttention(hidden_size)
    elif args.attention_type=="dot":
        attention = DotAttention
    else:
        attention = DotAttention

    if args.decoder_type=="baseline":
        decoder_rnn = DecoderRNNCell
    elif args.decoder_type=="joint":
        print("\n\nChoosing RNNCellJointCopy\n\n")
        decoder_rnn = DecoderRNNCellJointCopy
    else:
        decoder_rnn = DecoderRNNCell
    
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
    
    print(f"loading checkpoint from {checkpoint_dir}")
    print(f"latest checkpoint in the dir is {tf.train.latest_checkpoint(checkpoint_dir)}")
    checkpoint = tf.train.Checkpoint( model=model )
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print(status.assert_existing_objects_matched())

    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    for data, file_prefix in [(test_dataset, "test_"), (val_dataset, "val_")]:
        if file_prefix == "test_" and args.with_cp:
            cp_size = max_cp_size
        else:
            cp_size = None
        generate( model
                , data
                , file_prefix
                , args.output_path
                , eos
                , ix_to_tk
                , use_beam_search=args.beam_search
                , beam_size=args.beam_size
                , max_cp_size=cp_size
                , csap_model=args.with_cp)
    

if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
