from .model import Encoder
from .layers import DecoderRNNCell, DecoderRNNCellJointCopy, DotAttention, ConcatAttention
from .evaluation import evaluate

import numpy as np
import tensorflow as tf
import time
import os
import sys

def loss_function( x
                 , y
                 , loss_object
                 , train_scc_metrics
                 , train_acc_metrics):
    """ use only the non-pad values """
    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss_object(y, x)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    train_scc_metrics.update_state(y, x, sample_weight=mask)
    train_acc_metrics.update_state(y, x, sample_weight=mask)
    return tf.reduce_mean(loss_)

class TrainStepWrapper:
    @tf.function
    def train_step( self
                  , batch_data
                  , loss_object
                  , optimizer
                  , encoder
                  , decoderRNNCell
                  , train_acc_metrics
                  , train_scc_metrics
                  , last_out
                  , initial_state=None
                  , scheduled_sampling_rate : float = 0.5
                  , return_state_n=None):
        print("tracing tf.function with args:", file=sys.stderr)
        print(f"len(batch_data) : {len(batch_data)}", file=sys.stderr)
        for ix, d in enumerate(batch_data):
            print(f"batch_data[{ix}].shape : {d.shape}", file=sys.stderr)
        print(f"last_out.shape : {last_out.shape}", file=sys.stderr)
        print(f"last_out.dtype : {last_out.dtype}", file=sys.stderr)
        if initial_state is None:
          print(f"initial_state is None", file=sys.stderr)
        else:
          for ix, d in enumerate(initial_state):
            print(f"initial_state[{ix}].shape : {d.shape}", file=sys.stderr)
        loss = 0
        dec_in, targets, gen_or_teach, *tables = batch_data
        final_state = None
        final_last_out = None
        with tf.GradientTape() as tape:
            enc_outs, *last_hidden_rnn = encoder(tables)
            if initial_state is None:
                initial_state = [ last_hidden_rnn[-1]
                                , *last_hidden_rnn ]
            
            if isinstance(decoderRNNCell, DecoderRNNCellJointCopy):
                print("using joint copy mechanism !")
                enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), decoderRNNCell._word_vocab_size)
                aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
            else:
                print("using vanilla attention")
                aux_inputs = (enc_outs,)
            print(f"---\n", file=sys.stderr) 

            states = initial_state
            for t in range(dec_in.shape[1]):
                if (return_state_n is not None) and (t == return_state_n):
                    final_state = states
                    final_last_out = last_out
                if gen_or_teach[t] > scheduled_sampling_rate:
                    _input = last_out
                else:
                    _input = dec_in[:, t, :]
                last_out, states = decoderRNNCell( (_input, *aux_inputs)
                                                 , states=states
                                                 , training=True)
                loss += loss_function( last_out
                                     , targets[:, t]
                                     , loss_object
                                     , train_scc_metrics
                                     , train_acc_metrics)
                last_out = tf.expand_dims(tf.cast(tf.argmax(last_out, axis=1), tf.int16), -1)

        batch_loss = loss / int(targets.shape[1])
        variables = [ var for var in encoder.trainable_variables if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0')] + decoderRNNCell.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        if (return_state_n is None) or (return_state_n == dec_in.shape[1]):
            final_state = states
            final_last_out = last_out

        return batch_loss, final_state, final_last_out

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
    if learning_rate is None:
        optimizer = tf.keras.optimizers.Adam()
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                               , reduction='none')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( encoder=encoder
                                    , decoderRNNCell=decoderRNNCell)
    if load_last:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train_accurracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    train_scc_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    tsw = TrainStepWrapper()
    _generator = tf.random.Generator.from_non_deterministic_state()
    last_val_loss = None
    for epoch in range(epochs):
        start_time = time.time()
        print(f"-- started {epoch + 1}. epoch", flush=True)

        total_loss = 0

        for (num, batch_data) in enumerate(train_dataset.take(train_steps_per_epoch)):
            summaries, *tables = batch_data
            sums = tf.expand_dims(summaries, axis=-1)
            last_out = sums[:, 0]
            start = 0
            length = summaries.shape[1]
            state = None
            for end in range(truncation_size, length-1, truncation_skip_step):
                gen_or_teach = np.zeros(shape=(end-start))
                for i in range(len(gen_or_teach)):
                    gen_or_teach[i] = _generator.uniform(shape=(), maxval=1.0)
                truncated_data = ( sums[:, start:end, :]
                                 , summaries[:, start+1:end+1]
                                 , tf.convert_to_tensor(gen_or_teach)
                                 , *tables)
                batch_loss, state, last_out =  tsw.train_step( truncated_data
                                                             , loss_object
                                                             , optimizer
                                                             , encoder
                                                             , decoderRNNCell
                                                             , train_accurracy_metrics
                                                             , train_scc_metrics
                                                             , last_out
                                                             , initial_state=state
                                                             , scheduled_sampling_rate=scheduled_sampling_rate
                                                             , return_state_n=truncation_skip_step)
                total_loss += batch_loss
                start += truncation_skip_step
            if length % truncation_skip_step != 0:
                gen_or_teach = np.zeros(shape=(length-1-start))
                for i in range(len(gen_or_teach)):
                    gen_or_teach[i] = _generator.uniform(shape=(), maxval=1.0)
                truncated_data = ( sums[:, start:length-1, :]
                                 , summaries[:, start+1:length]
                                 , tf.convert_to_tensor(gen_or_teach)
                                 , *tables)
                batch_loss, state, last_out = tsw.train_step( truncated_data
                                                            , loss_object
                                                            , optimizer
                                                            , encoder
                                                            , decoderRNNCell
                                                            , train_accurracy_metrics
                                                            , train_scc_metrics
                                                            , last_out
                                                            , initial_state=state
                                                            , scheduled_sampling_rate=scheduled_sampling_rate)
                total_loss += batch_loss

            if num % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {num} Loss {train_scc_metrics.result():.4f}'+
                      f' Accurracy {train_accurracy_metrics.result()}', flush=True)
            

        # saving the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Epoch {epoch + 1} duration : {time.time() - start_time}", flush=True)
        if val_dataset is not None:
            final_val_loss = evaluate( val_dataset
                                     , val_steps
                                     , batch_size
                                     , ix_to_tk
                                     , val_save_path
                                     , eos
                                     , encoder
                                     , decoderRNNCell)
            if learning_rate is not None:
                if (last_val_loss is not None) and (final_val_loss > (last_val_loss + 0.005)):
                    optimizer.learning_rate = 0.5 * optimizer.learning_rate
                    print(f"halving the optimizer.learning rate to {optimizer.learning_rate}")
                last_val_loss = final_val_loss
            
        train_accurracy_metrics.reset_states()
        train_scc_metrics.reset_states()
