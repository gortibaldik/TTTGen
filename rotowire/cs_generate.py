from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from baseline_model.training import train
from baseline_model.layers import DotAttention, ConcatAttention, DecoderRNNCell, DecoderRNNCellJointCopy
from baseline_model.model import Encoder, EncoderDecoderBasic
from baseline_model.evaluation import evaluate
from argparse import ArgumentParser
import os
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
    return parser

def generate( model
            , dataset
            , file_prefix
            , dir_path
            , eos
            , ix_to_tk):
    predictions = model.predict(dataset)
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
    test_dataset, test_steps, *dummies = load_tf_record_dataset( test_path
                                                               , vocab_path
                                                               , batch_size
                                                               , False # shuffle
                                                               , max_table_size
                                                               , max_summary_size
                                                               , max_cp_size
                                                               , args.with_cp)
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
    model = EncoderDecoderBasic(encoder, decoderRNNCell)
    # compile the model - enables eager execution (my custom change)
    model.compile( tf.keras.optimizers.Adam()
                 , tf.keras.losses.SparseCategoricalCrossentropy()
                 , 1.0, 100, 50) # just some dummy values
    
    print(f"loading checkpoint from {checkpoint_dir}")
    print(f"latest checkpoint in the dir is {tf.train.latest_checkpoint(checkpoint_dir)}")
    checkpoint = tf.train.Checkpoint( model=model )
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print(status.assert_existing_objects_matched())

    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    for data, file_prefix in [(val_dataset, "val_"), (test_dataset, "test_")]:
        generate( model
                , data
                , file_prefix
                , args.output_path
                , eos
                , ix_to_tk)
    

if __name__ == "__main__":
    parser = _create_parser()
    _main(parser.parse_args())
