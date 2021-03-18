import tensorflow as tf
import time


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
                  , vocab_index_start
                  , BATCH_SIZE):
        loss = 0
        with tf.GradientTape() as tape:
            summaries, tables, fields, pos, rpos = batch_data
            outputs, (h, c), field_pos_embeddings = encoder((tables, fields, pos, rpos))
            decoder.initialize_batch(outputs, field_pos_embeddings, h, c)
            dec_input = tf.expand_dims([vocab_index_start] * BATCH_SIZE, 1)

            # go over all the time steps
            for t in range(0, summaries.shape[1]):
                predictions, attention_vector = decoder(dec_input)
                loss += loss_function( predictions
                                     , summaries[:, t]
                                     , loss_object)

                # teacher forcing
                dec_input = tf.expand_dims(summaries[:, t], 1)

        batch_loss = loss / int(summaries.shape[1])
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


def train( dataset
         , encoder
         , decoder
         , loss_object
         , optimizer
         , n_epochs
         , batch_size
         , steps_per_epoch
         , vocab_index_start
         , checkpoint
         , checkpoint_prefix):
    tsw = TrainStepWrapper()
    for epoch in range(n_epochs):
        start = time.time()

        total_loss = 0

        for (num, batch_data) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = tsw.train_step( batch_data
                                       , loss_object
                                       , optimizer
                                       , encoder
                                       , decoder
                                       , vocab_index_start
                                       , batch_size)
            total_loss += batch_loss

            if num % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {num} Loss {batch_loss.numpy():.4f}', flush=True)

        # saving the model every epoch
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Epoch {epoch + 1} duration : {time.time() - start}", flush=True)
