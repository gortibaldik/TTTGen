from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from baseline_model.training import train
from baseline_model.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from argparse import ArgumentParser
import os

def _create_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--word_emb_dim', type=int, default=128)
    parser.add_argument('--tp_emb_dim', type=int, default=128)
    parser.add_argument('--ha_emb_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--attention_type', type=str, default="dot")
    parser.add_argument('--decoder_type', type=str, default="joint")
    parser.add_argument('--truncation_size', type=int, default=100)
    parser.add_argument('--truncation_skip_step', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--scheduled_sampling_rate', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--with_cp', action='store_true')
    parser.add_argument('--manual', action='store_true')
    return parser

def _main(args):
    config_path = os.path.join(args.path, "config.txt")
    train_path = os.path.join(args.path, "train.tfrecord")
    valid_path = os.path.join(args.path, "valid.tfrecord")
    vocab_path = os.path.join(args.path, "all_vocab.txt")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    max_table_size, max_summary_size, max_cp_size = \
        load_values_from_config(config_path, load_cp=args.with_cp) # pylint: disable=unused-variable
    batch_size = args.batch_size
    dataset, steps, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=train_path # pylint: disable=unused-variable
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=True
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size
                                    , preprocess_cp_size=max_cp_size
                                    , with_content_plans=args.with_cp)
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

    if args.attention_type=="concat":
        attention = lambda: ConcatAttention(hidden_size)
    elif args.attention_type=="dot":
        attention = DotAttention
    else:
        attention = DotAttention

    if args.decoder_type=="baseline":
        decoder_rnn = DecoderRNNCell
    elif args.decoder_type=="joint":
        decoder_rnn = DecoderRNNCellJointCopy
    else:
        decoder_rnn = DecoderRNNCell

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
         , args.path
         , ix_to_tk
         , val_dataset
         , load_last=False
         , use_content_selection=args.with_cp
         , max_table_size=max_table_size
         , manual_training=args.manual)

if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
