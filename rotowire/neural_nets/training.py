from .encoders import Encoder, EncoderCS, EncoderCSBi
from .baseline_model import EncoderDecoderBasic
from .cp_model import EncoderDecoderContentSelection
from .layers import DecoderRNNCell, DecoderRNNCellJointCopy, DotAttention, ConcatAttention, \
                    ContentPlanDecoderCell
from .callbacks import CalcBLEUCallback, SaveOnlyModelCallback

import numpy as np
import tensorflow as tf
import time
import os
import sys

def create_basic_model( batch_size
                      , word_emb_dim
                      , word_vocab_size
                      , tp_emb_dim
                      , tp_vocab_size
                      , ha_emb_dim
                      , ha_vocab_size
                      , entity_span
                      , hidden_size
                      , attention_type
                      , decoderRNNInit
                      , dropout_rate
                      , encoder_cs_flag : bool = False
                      , encoder_cs_bidir_flag : bool = False
                      , max_table_size : int = None):
    if encoder_cs_flag and encoder_cs_bidir_flag :
        raise RuntimeError("Cannot choose both EncoderCS and EncoderCSBi as the encoder!")
    if encoder_cs_flag:
        encoder = EncoderCS( word_vocab_size
                           , word_emb_dim
                           , tp_vocab_size
                           , tp_emb_dim
                           , ha_vocab_size
                           , ha_emb_dim
                           , max_table_size
                           , hidden_size
                           , attention_type
                           , batch_size)
    elif encoder_cs_bidir_flag:
        encoder = EncoderCSBi( word_vocab_size
                             , word_emb_dim
                             , tp_vocab_size
                             , tp_emb_dim
                             , ha_vocab_size
                             , ha_emb_dim
                             , max_table_size
                             , hidden_size
                             , attention_type
                             , batch_size
                             , dropout_rate)
    else:
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
    return model

def create_cs_model( batch_size
                   , max_table_size
                   , word_emb_dim
                   , word_vocab_size
                   , tp_emb_dim
                   , tp_vocab_size
                   , ha_emb_dim
                   , ha_vocab_size
                   , hidden_size
                   , attention_type
                   , decoderRNNInit
                   , dropout_rate):
    encoder = EncoderCS( word_vocab_size
                       , word_emb_dim
                       , tp_vocab_size
                       , tp_emb_dim
                       , ha_vocab_size
                       , ha_emb_dim
                       , max_table_size
                       , hidden_size
                       , attention_type
                       , batch_size)
    cp_decoding_cell = ContentPlanDecoderCell( hidden_size
                                             , DotAttention
                                             , batch_size
                                             , dropout_rate)
    encoder_from_cp = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM( hidden_size
                            , return_sequences=True
                            , return_state=True
                            , dropout=dropout_rate),
        merge_mode='sum'
    )
    decoder_text = decoderRNNInit( word_vocab_size
                                 , word_emb_dim
                                 , hidden_size
                                 , batch_size
                                 , attention_type
                                 , dropout_rate)
    model = EncoderDecoderContentSelection( encoder
                                          , cp_decoding_cell
                                          , encoder_from_cp
                                          , decoder_text)
    return model


def train( train_dataset
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
         , load_last : bool = False
         , use_content_selection : bool = False
         , cp_training_rate : float = 0.2
         , max_table_size : int = None
         , manual_training : bool = True
         , encoder_cs_flag : bool = False
         , encoder_cs_bidir_flag : bool = False):

    if truncation_skip_step > truncation_size:
        raise RuntimeError(f"truncation_skip_step ({truncation_skip_step}) shouldn't be bigger"+
                           f"truncation_size ({truncation_size})")

    if not use_content_selection:
        model = create_basic_model( batch_size
                                  , word_emb_dim
                                  , word_vocab_size
                                  , tp_emb_dim
                                  , tp_vocab_size
                                  , ha_emb_dim
                                  , ha_vocab_size
                                  , entity_span
                                  , hidden_size
                                  , attention_type
                                  , decoderRNNInit
                                  , dropout_rate
                                  , encoder_cs_flag=encoder_cs_flag
                                  , encoder_cs_bidir_flag=encoder_cs_bidir_flag
                                  , max_table_size=max_table_size)
    else:
        model = create_cs_model( batch_size
                               , max_table_size
                               , word_emb_dim
                               , word_vocab_size
                               , tp_emb_dim
                               , tp_vocab_size
                               , ha_emb_dim
                               , ha_vocab_size
                               , hidden_size
                               , attention_type
                               , decoderRNNInit
                               , dropout_rate)
    
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( model=model)
    if load_last:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print(status.assert_existing_objects_matched())

    if not use_content_selection:
        model.compile( optimizer_1
                     , tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                                    , reduction='none')
                     , scheduled_sampling_rate
                     , truncation_size
                     , truncation_skip_step)
    else:
        model.compile( optimizer_1 # pylint: disable=too-many-function-args, unexpected-keyword-arg
                     , optimizer_2
                     , tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True # cp loss
                                                                    , reduction='none')
                     , tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False # dec loss
                                                                    , reduction='none')
                     , scheduled_sampling_rate
                     , truncation_size
                     , truncation_skip_step
                     , cp_training_rate=cp_training_rate)
    # define callbacks
    tensorboard_dir = os.path.join(checkpoint_dir, 'tb_logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    saving_callback = SaveOnlyModelCallback(checkpoint, checkpoint_prefix)

    bleu_callback = CalcBLEUCallback( val_dataset.take(5)
                                    , ix_to_tk
                                    , eos)
    # train
    if not manual_training:
        model.fit( train_dataset
                 , epochs=epochs
                 , callbacks=[tensorboard_callback, saving_callback, bleu_callback]
                 , validation_data=val_dataset)
    else:
        bleu_callback.set_model(model)
        cp_acc = []
        cp_loss = []
        txt_acc = []
        txt_loss = []
        for epoch in range(epochs):
            for ix, batch_data in enumerate(train_dataset):
                dct = model.train_step(batch_data)
                txt_acc.append(dct['accuracy_decoder'].numpy())
                cp_acc.append(dct['accuracy_cp'].numpy())
                txt_loss.append(dct['loss_decoder'].numpy())
                cp_loss.append(dct['loss_cp'].numpy())
                if ix % 10 == 0:
                    print(f"acc dec : {txt_acc[-1]:4.4f};" +\
                          f" acc cp : {cp_acc[-1]:4.4f};" +\
                          f" loss dec : {txt_loss[-1]:4.4f};" +\
                          f" loss cp: {cp_loss[-1]:4.4f}", flush=True)
            for metric in model.metrics:
                metric.reset_states()
            bleu_callback.on_epoch_end(epoch)
            saving_callback.on_epoch_end(epoch)

            for batch_data in val_dataset:
                dct = model.test_step(batch_data)
                print(f"acc dec : {dct['accuracy_decoder'].numpy()}; acc cp : {dct['accuracy_cp'].numpy()}; loss dec : {dct['loss_decoder'].numpy()}; loss cp: {dct['loss_cp'].numpy()}")
