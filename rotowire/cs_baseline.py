from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from baseline_model.training import train
from baseline_model.layers import DotAttention, ConcatAttention
from argparse import ArgumentParser
import os

def _create_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--word_emb_dim', type=int, default=600)
    parser.add_argument('--tp_emb_dim', type=int, default=600)
    parser.add_argument('--ha_emb_dim', type=int, default=600)
    parser.add_argument('--hidden_size', type=int, default=600)
    parser.add_argument('--attention_type', type=str, default="concat")
    parser.add_argument('--epochs', type=int, default=50)
    return parser

def _main(args):
    config_path = os.path.join(args.path, "config.txt")
    train_path = os.path.join(args.path, "train.tfrecord")
    valid_path = os.path.join(args.path, "valid.tfrecord")
    vocab_path = os.path.join(args.path, "all_vocab.txt")
    max_table_size, max_summary_size = load_values_from_config(config_path)
    batch_size = args.batch_size
    dataset, steps, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=train_path
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=True
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size)
    val_dataset, val_steps, *dummies = load_tf_record_dataset( valid_path
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

    checkpoint_dir = os.path.join(args.path, "training_checkpoints/")

    if args.attention_type="concat":
        attention = ConcatAttention
    elif args.attention_type="dot":
        attention = DotAttention
    else:
        attention = ConcatAttention

    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    train( dataset
         , steps
         , checkpoint_dir
         , batch_size
         , max_summary_size - 1
         , word_emb_dim
         , word_vocab_size
         , tp_emb_dim
         , tp_vocab_size
         , ha_emb_dim
         , ha_vocab_size
         , entity_span
         , hidden_size
         , 1
         , 5
         , eos
         , attention_type=attention
         , val_save_path=args.path
         , ix_to_tk=ix_to_tk
         , val_dataset=val_dataset
         , val_steps=val_steps
         , load_last=False)


if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())