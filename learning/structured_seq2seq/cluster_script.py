from vocab import Vocab
from model import Encoder, Decoder
from training import train
from evaluation import evaluate

import data_loading
import tensorflow as tf
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model"
    )
    parser.add_argument(
        '--task_dir',
        default="/home/trebuna/seq2seq_structure_aware",
        type=str,
        help='directory with all the data needed for training and validation'
    )
    args = parser.parse_args()
    task_dir = args.task_dir
    prefix = os.path.join(task_dir, "processed_data")
    print(prefix)
    data_loading.MODEL_SET_DATA_DIR = prefix
    fields_path = os.path.join(prefix,"field_vocab.txt")
    words_path = os.path.join(prefix, "word_vocab.txt")
    vocab = Vocab(fields_path, words_path)
    batch_size = 32
    num_examples = 5000

    data, steps_per_epoch = data_loading.load_dataset( data_loading.ModelSet.val
                                                     , batch_size=batch_size
                                                     , shuffle=False
                                                     , num_examples=num_examples)

    print("Training dataset loaded!")
    words_emb_dim = 400
    fields_emb_dim = 50
    pos_emb_dim = 5
    pos_vocab_size = 31
    hidden_dim = 500
    encoder = Encoder( vocab.get_words_size()
                     , words_emb_dim
                     , vocab.get_fields_size()
                     , fields_emb_dim
                     , pos_vocab_size
                     , pos_emb_dim
                     , hidden_dim)

    decoder = Decoder( vocab.get_words_size()
                     , words_emb_dim
                     , encoder.get_field_pos_emb_dim()
                     , hidden_dim)

    learning_rate = 0.0003
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none')
    print("Model created!")

    # create checkpoints
    checkpoint_dir = os.path.join(task_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer
                                     , encoder=encoder
                                     , decoder=decoder)

    val_data, val_steps_per_epoch = data_loading.load_dataset( data_loading.ModelSet.val
                                                             , batch_size=batch_size
                                                             , shuffle=False
                                                             , num_examples=num_examples)

    n_improvements = 10
    for i in range(n_improvements):
        n_epochs = 5
        train( data
             , encoder
             , decoder
             , loss_object
             , optimizer
             , n_epochs
             , batch_size
             , steps_per_epoch
             , vocab.word2id(Vocab.START_TOKEN)
             , checkpoint
             , checkpoint_prefix)

        gold_path_prefix = os.path.join(task_dir, "processed_data", "valid", "valid_split_for_rouge" , "gold_summary_")
        evaluate( val_data
                , data_loading.ModelSet.val
                , val_steps_per_epoch
                , batch_size
                , vocab
                , gold_path_prefix
                , os.path.join(prefix, "predictions_no_attention_" + str(i) + ".txt")
                , os.path.join(prefix, "predictions_with_attention" + str(i) + ".txt")
                , encoder
                , decoder)
