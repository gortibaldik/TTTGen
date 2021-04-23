from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from baseline_model.training import train
from baseline_model.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from baseline_model.model import Encoder
from baseline_model.evaluation import evaluate
from argparse import ArgumentParser
import os
import tensorflow as tf

def _create_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--word_emb_dim', type=int, default=128)
    parser.add_argument('--tp_emb_dim', type=int, default=128)
    parser.add_argument('--ha_emb_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--attention_type', type=str, default="dot")
    parser.add_argument('--decoder_type', type=str, default="joint")
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    return parser

def _main(args):
    config_path = os.path.join(args.path, "config.txt")
    valid_path = os.path.join(args.path, "valid.tfrecord")
    test_path = os.path.join(args.path, "test.tfrecord")
    vocab_path = os.path.join(args.path, "all_vocab.txt")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    max_table_size, max_summary_size = load_values_from_config(config_path)
    batch_size = args.batch_size
    val_dataset, val_steps, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=valid_path
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=False
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size)
    test_dataset, test_steps, *dummies = load_tf_record_dataset( test_path
                                                               , vocab_path
                                                               , batch_size
                                                               , False # shuffle
                                                               , max_table_size
                                                               , max_summary_size)
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
    
    encoder = Encoder( word_vocab_size
                     , word_emb_dim
                     , tp_vocab_size
                     , tp_emb_dim
                     , ha_vocab_size
                     , ha_emb_dim
                     , entity_span
                     , hidden_size
                     , batch_size)
    
    decoderRNNCell = decoder_rnn( word_vocab_size
                                , word_emb_dim
                                , hidden_size
                                , batch_size
                                , attention=attention
                                , dropout_rate=args.dropout_rate)
    
    print(f"loading checkpoint from {checkpoint_dir}")
    print(f"latest checkpoint in the dir is {tf.train.latest_checkpoint(checkpoint_dir)}")
    checkpoint_prefix = os.path.join( checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( encoder=encoder
                                    , decoderRNNCell=decoderRNNCell)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    final_val_loss = evaluate( val_dataset
                             , val_steps
                             , batch_size
                             , ix_to_tk
                             , "."
                             , eos
                             , encoder
                             , decoderRNNCell
                             , bleu_batches=1) # each batch should be used for generation
    final_val_loss = evaluate( test_dataset
                             , test_steps
                             , batch_size
                             , ix_to_tk
                             , "."
                             , eos
                             , encoder
                             , decoderRNNCell
                             , bleu_batches=1) # each batch should be used for generation


if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
