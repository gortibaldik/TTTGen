from .model import Encoder
from .layers import DecoderRNNCell, DotAttention, ConcatAttention
from .evaluation import evaluate
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
                  , initial_state=None):
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
        print(f"---\n", file=sys.stderr)
        loss = 0
        dec_in, targets, *tables = batch_data
        with tf.GradientTape() as tape:
            enc_outs, *last_hidden_rnn = encoder(tables)
            if initial_state is None:
                initial_state = [ last_hidden_rnn[-1]
                                , *last_hidden_rnn ]

            states = initial_state
            for t in range(dec_in.shape[1]):
                last_out, states = decoderRNNCell( (dec_in[:, t, :], enc_outs)
                                                 , states=states
                                                 , training=True)
                loss += loss_function( last_out
                                     , targets[:, t]
                                     , loss_object
                                     , train_scc_metrics
                                     , train_acc_metrics)

        batch_loss = loss / int(targets.shape[1])
        variables = [ var for var in encoder.trainable_variables if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0')] + decoderRNNCell.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, states, last_out

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
         , truncation_size
         , dropout_rate
         , attention_type=DotAttention
         , val_save_path : str = None
         , ix_to_tk : dict = None
         , val_dataset = None
         , val_steps = None
         , load_last : bool = False):
    encoder = Encoder( word_vocab_size
                     , word_emb_dim
                     , tp_vocab_size
                     , tp_emb_dim
                     , ha_vocab_size
                     , ha_emb_dim
                     , entity_span
                     , hidden_size
                     , batch_size)
    decoderRNNCell = DecoderRNNCell( word_vocab_size
                                   , word_emb_dim
                                   , hidden_size
                                   , batch_size
                                   , attention=attention_type
                                   , dropout_rate=dropout_rate)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                               , reduction='none')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( optimizer=optimizer
                                    , encoder=encoder
                                    , decoderRNNCell=decoderRNNCell)
    if load_last:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train_accurracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    train_scc_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    tsw = TrainStepWrapper()
    for epoch in range(epochs):
        start_time = time.time()
        print(f"-- started {epoch + 1}. epoch", flush=True)

        total_loss = 0

        for (num, batch_data) in enumerate(train_dataset.take(train_steps_per_epoch)):
            summaries, *tables = batch_data
            sums = tf.expand_dims(summaries, axis=-1)
            last_out = start=tf.one_hot( tf.cast( summaries[:, 0]
                                                          , tf.int32)
                                                 , word_vocab_size
                                                 , axis=-1)
            start = 0
            length = summaries.shape[1]
            state = None
            for end in range(truncation_size, length-1, truncation_size):
                truncated_data = (sums[:, start:end, :], summaries[:, start+1:end+1], *tables)
                batch_loss, state, last_out =  tsw.train_step( truncated_data
                                                             , loss_object
                                                             , optimizer
                                                             , encoder
                                                             , decoderRNNCell
                                                             , train_accurracy_metrics
                                                             , train_scc_metrics
                                                             , last_out
                                                             , initial_state=state)
                total_loss += batch_loss
                start = end
            if length % truncation_size != 0:
                truncated_data = (sums[:, start:length-1, :], summaries[:, start+1:length], *tables)
                batch_loss, state, last_out = tsw.train_step( truncated_data
                                                            , loss_object
                                                            , optimizer
                                                            , encoder
                                                            , decoderRNNCell
                                                            , train_accurracy_metrics
                                                            , train_scc_metrics
                                                            , last_out
                                                            , initial_state=state)
                total_loss += batch_loss

            if num % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {num} Loss {train_scc_metrics.result():.4f}'+
                      f' Accurracy {train_accurracy_metrics.result()}', flush=True)
            

        # saving the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Epoch {epoch + 1} duration : {time.time() - start_time}", flush=True)
        evaluate( val_dataset
                , val_steps
                , batch_size
                , ix_to_tk
                , val_save_path
                , eos
                , encoder
                , decoderRNNCell)
        train_accurracy_metrics.reset_states()
        train_scc_metrics.reset_states()