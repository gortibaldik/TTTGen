from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from neural_nets.training import train
from neural_nets.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from argparse import ArgumentParser
import os

def _create_parser():
    parser = ArgumentParser()
    parser.add_argument( '--path'
                       , type=str
                       , help="Path to prepared dataset files"
                       , required=True)
    parser.add_argument( '--batch_size'
                       , type=int
                       , default=16)
    parser.add_argument( '--word_emb_dim'
                       , type=int
                       , help="embedding dimensionality to which to project the words from the summary " +\
                              "and value and entity part of the input records"
                       , default=128)
    parser.add_argument( '--tp_emb_dim'
                       , type=int
                       , help="embedding dimensionality to which to project the type part of " +\
                              "input record"
                       , default=128)
    parser.add_argument( '--ha_emb_dim'
                       , type=int
                       , help="embedding dimensionality to which project the home/away flag part of " +\
                              "input record"
                       , default=128)
    parser.add_argument( '--hidden_size'
                       , help="dimensionality of the hidden states"
                       , type=int
                       , default=300)
    parser.add_argument( '--attention_type'
                       , type=str
                       , help="one of [dot, concat], specifies whether to use Luong style dot attention " +\
                              "or Bahdanau style concat attention"
                       , default="dot")
    parser.add_argument( '--decoder_type'
                       , type=str
                       , help="one of [joint, baseline], specifies whether to use baseline decoder " +\
                              "or decoder with joint-copy mechanism"
                       , default="joint")
    parser.add_argument( '--truncation_size'
                       , type=int
                       , help="t_2 argument of TBPTT (how long to run BPTT)"
                       , default=100)
    parser.add_argument( '--truncation_skip_step'
                       , type=int
                       , help="t_1 argument of TBPTT (should be lower than or equal to t_2)"
                       , default=50)
    parser.add_argument( '--epochs'
                       , type=int
                       , help="number of training epochs"
                       , default=50)
    parser.add_argument( '--dropout_rate'
                       , type=float
                       , help="rate at which to drop cells and corresponding connections at " +\
                              "the outputs of the internal LSTM layers of the model (number " +\
                              "between 0 and 1, 0 means no dropout)"
                       , default=0.2)
    parser.add_argument( '--scheduled_sampling_rate'
                       , type=float
                       , help="frequency at which the gold outputs from the previous time-steps " +\
                               "are fed into the network (number between 0 and 1, 1 means regular " +\
                               "training)"
                       , default=1.0)
    parser.add_argument( '--learning_rate'
                       , type=float
                       , help="learning_rate argument for the Adam optimizer"
                       , default=0.001)
    parser.add_argument( '--with_cp'
                       , help="train EncoderDecoderContentSelection, with content planning"
                       , action='store_true')
    parser.add_argument( '--cp_training_rate'
                       , type=float
                       , help="number between 0 and 1, fraction of batches where we also train " +\
                              "the content planning decoder"
                       , default=0.2)
    parser.add_argument( '--manual'
                       , help="instead of training using tf.keras.Model.fit() use manual loops"
                       , action='store_true')
    parser.add_argument( '--load_last'
                       , help="load last saved model from ${args.path}/training_checkpoints" 
                       , action='store_true')
    parser.add_argument( '--with_cs'
                       , help="create EncoderDecoderBaseline with Content Selection Encoder (CopyCS model)" 
                       , action='store_true')
    parser.add_argument( '--with_csbidir'
                       , help="create EncoderDecoderBaseline with Content Selection Encoder and " +\
                              "bidirectional LSTM over it (CopyCSBidir model)"
                       , action='store_true')
    return parser

def _main(args):
    config_path = os.path.join(args.path, "config.txt")
    train_path = os.path.join(args.path, "train.tfrecord")
    valid_path = os.path.join(args.path, "valid.tfrecord")
    vocab_path = os.path.join(args.path, "all_vocab.txt")
    for key, value in vars(args).items():
        print(f"{key} : {value}")

    # load config file
    max_table_size, max_summary_size, max_cp_size = \
        load_values_from_config(config_path, load_cp=args.with_cp) # pylint: disable=unused-variable
    batch_size = args.batch_size

    # load training dataset
    dataset, steps, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=train_path # pylint: disable=unused-variable
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=True
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size
                                    , preprocess_cp_size=max_cp_size
                                    , with_content_plans=args.with_cp)
    
    # load validation dataset
    val_dataset, val_steps, *dummies = load_tf_record_dataset( valid_path # pylint: disable=unused-variable
                                                             , vocab_path
                                                             , batch_size
                                                             , False # shuffle
                                                             , max_table_size
                                                             , max_summary_size
                                                             , preprocess_cp_size=max_cp_size
                                                             , with_content_plans=args.with_cp)
    word_vocab_size = len(tk_to_ix)
    word_emb_dim = args.word_emb_dim
    tp_vocab_size = len(tp_to_ix)
    tp_emb_dim = args.tp_emb_dim
    ha_vocab_size = len(ha_to_ix)
    ha_emb_dim = args.ha_emb_dim
    entity_span = 22
    hidden_size = args.hidden_size

    checkpoint_dir = os.path.join(args.path, "training_checkpoints/")

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
        decoder_rnn = DecoderRNNCellJointCopy
    else:
        decoder_rnn = DecoderRNNCell

    # run the train script
    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    train( dataset
         , checkpoint_dir
         , batch_size
         , word_emb_dim
         , word_vocab_size
         , tp_emb_dim
         , tp_vocab_size
         , ha_emb_dim
         , ha_vocab_size
         , entity_span
         , hidden_size
         , args.learning_rate
         , args.epochs
         , eos
         , args.dropout_rate
         , args.scheduled_sampling_rate
         , args.truncation_size
         , args.truncation_skip_step
         , attention
         , decoder_rnn
         , ix_to_tk
         , val_dataset
         , load_last=args.load_last
         , use_content_selection=args.with_cp
         , cp_training_rate=args.cp_training_rate
         , max_table_size=max_table_size
         , manual_training=args.manual
         , encoder_cs_flag=args.with_cs
         , encoder_cs_bidir_flag=args.with_csbidir)

if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
