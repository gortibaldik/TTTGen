from .model import Encoder, EncoderDecoderBasic
from .layers import DecoderRNNCell, DecoderRNNCellJointCopy, DotAttention, ConcatAttention
from .evaluation import evaluate
from .callbacks import CalcBLEUCallback, SaveOnlyModelCallback

import numpy as np
import tensorflow as tf
import time
import os
import sys

def train( train_dataset
         , train_steps_per_epoch
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
         , learning_rate
         , epochs
         , eos
         , dropout_rate
         , scheduled_sampling_rate
         , truncation_size
         , truncation_skip_step
         , attention_type=DotAttention
         , decoderRNNInit=DecoderRNNCell
         , val_save_path : str = None
         , ix_to_tk : dict = None
         , val_dataset = None
         , val_steps = None
         , load_last : bool = False):

    if truncation_skip_step > truncation_size:
        raise RuntimeError(f"truncation_skip_step ({truncation_skip_step}) shouldn't be bigger"+
                           f"truncation_size ({truncation_size})")

    encoder = Encoder( word_vocab_size
                     , word_emb_dim
                     , tp_vocab_size
                     , tp_emb_dim
                     , ha_vocab_size
                     , ha_emb_dim
                     , entity_span
                     , hidden_size
                     , batch_size)
    decoderRNNCell = decoderRNNInit( word_vocab_size
                                   , word_emb_dim
                                   , hidden_size
                                   , batch_size
                                   , attention=attention_type
                                   , dropout_rate=dropout_rate)
    model = EncoderDecoderBasic(encoder, decoderRNNCell)
    if learning_rate is None:
        optimizer = tf.keras.optimizers.Adam()
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                               , reduction='none')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( model=model)
    if load_last:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print(status.assert_existing_objects_matched())
    model.compile( optimizer
                 , loss_object
                 , scheduled_sampling_rate
                 , truncation_size
                 , truncation_skip_step)
    # define callbacks
    tensorboard_dir = os.path.join(checkpoint_dir, 'tb_logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    saving_callback = SaveOnlyModelCallback(checkpoint, checkpoint_prefix)

    bleu_callback = CalcBLEUCallback( val_dataset.take(5)
                                    , ix_to_tk
                                    , eos)
    # train
    model.fit( train_dataset
             , epochs=epochs
             , callbacks=[tensorboard_callback, saving_callback, bleu_callback]
             , validation_data=val_dataset)
