import tensorflow as tf
import numpy as np
import sys
from baseline_model.layers import Embedding, MLPEncodingCell, ContentSelectionCell, DecoderRNNCellJointCopy

class Encoder(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim
                , entity_span
                , hidden_size
                , batch_size):
        super(Encoder, self).__init__()
        self._embedding = Embedding( word_vocab_size
                                   , word_emb_dim
                                   , word_vocab_size
                                   , word_emb_dim
                                   , tp_vocab_size
                                   , tp_emb_dim
                                   , ha_vocab_size
                                   , ha_emb_dim)
        emb_size = word_emb_dim * 2 + tp_emb_dim + ha_emb_dim
        self._rnncell = MLPEncodingCell(hidden_size, emb_size)
        self._rnn = tf.keras.layers.RNN(self._rnncell, return_sequences=True, return_state=True)
        self._pooling = tf.keras.layers.AveragePooling1D(pool_size=entity_span)
        self._linear_transform = tf.keras.layers.Dense(4*hidden_size, use_bias=False, name="linear_transform")
        self._batch_size = batch_size

    def change_batch_size(self, new_batch_size : int):
        self._batch_size = new_batch_size

    def call(self, inputs):
        embedded = self._embedding(inputs)
        embedded = tf.concat(embedded, axis=2) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        all_states, _ = self._rnn(embedded)
        pooled = self._pooling(all_states)
        pooled = tf.reshape(pooled, [self._batch_size, -1])
        trans = self._linear_transform(pooled)
        hidden_1, cell_1, hidden_2, cell_2 = tf.split(trans, 4 , 1)
        return all_states, hidden_1, cell_1, hidden_2, cell_2

class EncoderCS(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim
                , max_seq_size
                , hidden_size
                , attention_type
                , batch_size):
        super(EncoderCS, self).__init__()
        self._embedding = Embedding( word_vocab_size
                                   , word_emb_dim
                                   , word_vocab_size
                                   , word_emb_dim
                                   , tp_vocab_size
                                   , tp_emb_dim
                                   , ha_vocab_size
                                   , ha_emb_dim)
        emb_size = word_emb_dim * 2 + tp_emb_dim + ha_emb_dim
        self._rnncell = MLPEncodingCell(hidden_size, emb_size)
        self._content_selection_cell = ContentSelectionCell(attention_type, max_seq_size, hidden_size, hidden_size)
        self._rnn = tf.keras.layers.RNN(self._rnncell, return_sequences=True, return_state=True)
        self._cs_rnn = tf.keras.layers.RNN(self._content_selection_cell, return_sequences=True, return_state=True)
        self._batch_size = batch_size

    def change_batch_size(self, new_batch_size : int):
        self._batch_size = new_batch_size

    def call(self, inputs):
        embedded = self._embedding(inputs)
        embedded = tf.concat(embedded, axis=2) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        all_states, _ = self._rnn(embedded)
        content_selected, *_ = self._cs_rnn(all_states, initial_state=(all_states, tf.zeros((), dtype=tf.int32)))
        avg = tf.reduce_mean(content_selected, axis=1)
        return content_selected, avg, avg, avg, avg

class EncoderDecoderBasic(tf.keras.Model):
    def __init__( self
                , encoder
                , decoder_cell):
        super(EncoderDecoderBasic, self).__init__()
        self._encoder = encoder
        self._decoder_cell = decoder_cell

    def compile(self
               , optimizer
               , loss_fn
               , scheduled_sampling_rate
               , truncation_size
               , truncation_skip_step):
        super(EncoderDecoderBasic, self).compile(run_eagerly=True)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._train_metrics = [ tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
                              , tf.keras.metrics.SparseCategoricalCrossentropy(name='loss')]
        self._val_metrics = { "val_accuracy" : tf.keras.metrics.Accuracy(name='accuracy')
                            , "val_loss" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss')}
        self._scheduled_sampling_rate = scheduled_sampling_rate
        self._truncation_skip_step = truncation_skip_step
        self._truncation_size = truncation_size
        self._generator = tf.random.Generator.from_non_deterministic_state()

    def _calc_loss( self, x, y):
        """ use only the non-pad values """
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss_ = self._loss_fn(y, x)
        mask = tf.cast(mask, loss_.dtype)
        loss_ *= mask
        for metric in self._train_metrics:
            metric.update_state(y, x, sample_weight=mask)
        return tf.reduce_mean(loss_)

    @tf.function
    def bppt_step( self
                 , batch_data
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
        loss = 0
        dec_in, targets, gen_or_teach, *tables = batch_data
        final_state = None
        final_last_out = None
        with tf.GradientTape() as tape:
            enc_outs, *last_hidden_rnn = self._encoder(tables)
            if initial_state is None:
                initial_state = [ last_hidden_rnn[-1]
                                , *last_hidden_rnn ]

            if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
                print("using joint copy mechanism !")
                enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
            else:
                print("using vanilla attention")
                aux_inputs = (enc_outs,)
            print(f"---\n", file=sys.stderr)

            states = initial_state
            for t in range(dec_in.shape[1]):
                if (self._truncation_skip_step is not None) and (t == self._truncation_skip_step):
                    final_state = states
                    final_last_out = last_out
                if gen_or_teach[t] > self._scheduled_sampling_rate:
                    _input = last_out
                else:
                    _input = dec_in[:, t, :]
                last_out, states = self._decoder_cell( (_input, *aux_inputs)
                                                     , states=states
                                                     , training=True)
                loss += self._calc_loss( last_out
                                       , targets[:, t])
                last_out = tf.expand_dims(tf.cast(tf.argmax(last_out, axis=1), tf.int16), -1)

        variables = []
        for var in self._encoder.trainable_variables + self._decoder_cell.trainable_variables:
            if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0'):
                variables.append(var)

        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))

        if (self._truncation_skip_step is None) or (self._truncation_skip_step == dec_in.shape[1]):
            final_state = states
            final_last_out = last_out

        return final_state, final_last_out


    @property
    def metrics(self):
        return self._train_metrics + list(self._val_metrics.values())


    def train_step(self, batch_data):
        summaries, *tables = batch_data
        sums = tf.expand_dims(summaries, axis=-1)
        last_out = sums[:, 0]
        start = 0
        length = summaries.shape[1]
        state = None
        for end in range(self._truncation_size, length-1, self._truncation_skip_step):
            gen_or_teach = np.zeros(shape=(end-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)
            truncated_data = ( sums[:, start:end, :]
                             , summaries[:, start+1:end+1]
                             , tf.convert_to_tensor(gen_or_teach)
                             , *tables)
            state, last_out =  self.bppt_step( truncated_data
                                             , last_out
                                             , initial_state=state)
            start += self._truncation_skip_step
        # finish the truncated bppt if the truncation_size cannot divide properly
        # the length of sequence
        if (length - self._truncation_size) % self._truncation_skip_step != 0:
            gen_or_teach = np.zeros(shape=(length-1-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)
            truncated_data = ( sums[:, start:length-1, :]
                                , summaries[:, start+1:length]
                                , tf.convert_to_tensor(gen_or_teach)
                                , *tables)
            state, last_out = self.bppt_step( truncated_data
                                            , last_out
                                            , initial_state=state)
        return dict([(metric.name, metric.result()) for metric in self._train_metrics])


    def test_step(self, batch_data):
        summaries, *tables = batch_data
        max_sum_size = summaries.shape[1] - 1
        dec_inputs = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]
        targets = summaries[:, 1:max_sum_size+1]

        enc_outs, *last_hidden_rnn = self._encoder(tables)

        if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (enc_outs,)

        initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]
        dec_in = dec_inputs[:, 0, :] # start tokens

        result_preds = np.zeros(targets.shape, dtype=np.int)

        for t in range(targets.shape[1]):
            pred, initial_state = self._decoder_cell( (dec_in, *aux_inputs)
                                                    , initial_state
                                                    , training=False)

            mask = tf.math.logical_not(tf.math.equal(targets[:, t], 0))
            self._val_metrics['val_loss'].update_state( targets[:, t]
                                                      , pred
                                                      , sample_weight=mask )

            predicted_ids = tf.argmax(pred, axis=1).numpy()
            result_preds[:, t] = predicted_ids
            dec_in = tf.expand_dims(targets[:, t], axis=1)

        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        self._val_metrics['val_accuracy'].update_state( targets
                                                      , result_preds
                                                      , sample_weight=mask)

        return dict([(metric.name, metric.result()) for metric in self._val_metrics.values()])

    def predict_step(self, data):
        summaries, *tables = data

        # retrieve start tokens
        dec_inputs = tf.expand_dims(summaries, axis=-1)
        dec_in = dec_inputs[:, 0, :] # start tokens

        enc_outs, *last_hidden_rnn = self._encoder(tables)

        if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (enc_outs,)

        initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]

        result_preds = np.zeros(summaries.shape, dtype=np.int)

        for t in range(summaries.shape[1]):
            pred, initial_state = self._decoder_cell( (dec_in, *aux_inputs)
                                                    , initial_state
                                                    , training=False)

            predicted_ids = tf.argmax(pred, axis=1).numpy()
            result_preds[:, t] = predicted_ids
            dec_in = tf.expand_dims(predicted_ids, axis=1)

        return result_preds
