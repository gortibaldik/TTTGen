from .model import Encoder, Decoder
import tensorflow as tf
import time
import os

def loss_function(x, y, loss_object):
    """
    only count those values which count in the computations
    - meaning, we ignore padding 0 values
    :return:
    """
    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss_object(y, x)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

class TrainStepWrapper:
    @tf.function
    def train_step( self
                  , batch_data
                  , loss_object
                  , optimizer
                  , encoder
                  , decoder
                  , BATCH_SIZE):
        loss = 0
        with tf.GradientTape() as tape:
            summaries, *tables = batch_data
            enc_outs, *last_hidden_rnn = encoder(tables)
            last_hidden_attn = last_hidden_rnn[-1]

            # go over all the time steps
            for t in range(0, summaries.shape[1] - 1):
                # teacher forcing for any index bigger than 0
                dec_input = tf.expand_dims(summaries[:, t], 1)
                preds, last_hidden_attn, align, *last_hidden_rnn = decoder( dec_input
                                                                          , last_hidden_rnn
                                                                          , last_hidden_attn
                                                                          , enc_outs)
                loss += loss_function( preds
                                     , summaries[:, t + 1]
                                     , loss_object)


        batch_loss = loss / int(summaries.shape[1])
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

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
         , epochs):
    encoder = Encoder( word_vocab_size
                     , word_emb_dim
                     , tp_vocab_size
                     , tp_emb_dim
                     , ha_vocab_size
                     , ha_emb_dim
                     , entity_span
                     , hidden_size
                     , batch_size)
    decoder = Decoder( word_vocab_size
                     , word_emb_dim
                     , hidden_size
                     , batch_size)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False
                                                               , reduction='none')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( optimizer=optimizer
                                    , encoder=encoder
                                    , decoder=decoder)
    tsw = TrainStepWrapper()
    for epoch in range(epochs):
        start = time.time()
        print(f"-- started {epoch + 1}. epoch", flush=True)

        total_loss = 0

        for (num, batch_data) in enumerate(train_dataset.take(train_steps_per_epoch)):
            batch_loss = tsw.train_step( batch_data
                                       , loss_object
                                       , optimizer
                                       , encoder
                                       , decoder
                                       , batch_size)
            total_loss += batch_loss

            if num % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {num} Loss {batch_loss.numpy():.4f}', flush=True)

        # saving the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Epoch {epoch + 1} duration : {time.time() - start}", flush=True)