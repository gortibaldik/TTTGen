from .model import Encoder
from .layers import DecoderRNNCell
import tensorflow as tf
import time
import os

def loss_function(x, y, loss_object, train_scc_metrics):
    """ use only the non-pad values """
    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss_object(y, x)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    train_scc_metrics.update_state(y, x, sample_weight=mask)
    return tf.reduce_mean(loss_)

class TrainStepWrapper:
    @tf.function
    def train_step( self
                  , batch_data
                  , loss_object
                  , optimizer
                  , encoder
                  , decoderRNNCell
                  , decoderRNN
                  , train_accurracy_metrics
                  , train_scc_metrics):
        loss = 0
        dec_input, targets, *tables = batch_data
        with tf.GradientTape() as tape:
            enc_outs, *last_hidden_rnn = encoder(tables)
            initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]
            decoderRNNCell.initialize_enc_outs(enc_outs)

            outputs = decoderRNN( dec_input, initial_state=initial_state)
            loss += loss_function( outputs
                                 , targets
                                 , loss_object
                                 , train_scc_metrics)

        batch_loss = loss
        variables = encoder.trainable_variables + decoderRNNCell.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        train_accurracy_metrics.update_state(targets, outputs, sample_weight=mask)

        return batch_loss

def train( train_dataset
         , train_steps_per_epoch
         , checkpoint_dir
         , batch_size
         , max_sum_size
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
                                   , batch_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                               , reduction='none')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( optimizer=optimizer
                                    , encoder=encoder
                                    , decoderRNNCell=decoderRNNCell)
    if load_last:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    decoderRNN = tf.keras.layers.RNN( decoderRNNCell
                                    , return_sequences=True)
    train_accurracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    train_scc_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    tsw = TrainStepWrapper()
    for epoch in range(epochs):
        start = time.time()
        print(f"-- started {epoch + 1}. epoch", flush=True)

        total_loss = 0

        for (num, batch_data) in enumerate(train_dataset.take(train_steps_per_epoch)):
            summaries, *tables = batch_data
            sums = tf.expand_dims(summaries, axis=-1)
            batch_data = (sums[:, :max_sum_size, :], summaries[:, 1:max_sum_size+1], *tables)
            batch_loss = tsw.train_step( batch_data
                                       , loss_object
                                       , optimizer
                                       , encoder
                                       , decoderRNNCell
                                       , decoderRNN
                                       , train_accurracy_metrics
                                       , train_scc_metrics)
            total_loss += batch_loss

            if num % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {num} Loss {train_scc_metrics.result():.4f}'+
                      f' Accurracy {train_accurracy_metrics.result()}', flush=True)

        # saving the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Epoch {epoch + 1} duration : {time.time() - start}", flush=True)
        evaluate( val_dataset
                , val_steps
                , batch_size
                , max_sum_size
                , ix_to_tk
                , val_save_path
                , encoder
                , decoderRNNCell)
        train_accurracy_metrics.reset_states()
        train_scc_metrics.reset_states()