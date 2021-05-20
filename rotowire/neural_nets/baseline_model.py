import tensorflow as tf
import numpy as np
import sys

from neural_nets.layers import DecoderRNNCellJointCopy

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


class BeamSearchAdapter(tf.keras.Model):
    def __init__( self
                , encoder_decoder_basic : EncoderDecoderBasic
                , beam_size
                , eos):
        super(BeamSearchAdapter, self).__init__()
        self._encoder = encoder_decoder_basic._encoder
        self._decoder_cell = encoder_decoder_basic._decoder_cell
        self._beam_size = beam_size
        self._eos = eos

    def compile(self):
        super(BeamSearchAdapter, self).compile(run_eagerly=True)

    def call(self, batch_data):
        summaries, *tables = batch_data

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

        actual_beam_size = self._beam_size
        hypotheses = []
        mask = tf.ones(shape=dec_in.shape, dtype=tf.bool)
        hypotheses.append((dec_in, mask, tf.zeros(shape=summaries.shape[0]), initial_state))

        out_sentence = tf.TensorArray(size=summaries.shape[1], dtype=tf.int32)
        out_predecessors = tf.TensorArray(size=summaries.shape[1], dtype=tf.int32)
        for t in range(summaries.shape[1]):
            bs_sqrd = len(hypotheses) * actual_beam_size
            scores = tf.TensorArray(size=bs_sqrd, dtype=tf.float32)
            indices = tf.TensorArray(size=bs_sqrd, dtype=tf.int32)
            prdcsrs = tf.TensorArray(size=bs_sqrd, dtype=tf.int32)
            initial_states = []
            masks = []
            siix = 0
            for h in range(len(hypotheses)):
                dec_in, mask, nlogsum, initial_state = hypotheses.pop(0)
                mask = tf.math.logical_not(tf.math.logical_or(tf.math.logical_not(mask) , dec_in == self._eos))
                pred, (hatt, h1, c1, h2, c2) = self._decoder_cell( (dec_in, *aux_inputs)
                                                                    , initial_state
                                                                    , training=False)
                top_predicted = tf.math.top_k(pred, k=actual_beam_size, sorted=True)
                ixs = top_predicted.indices * tf.cast(mask, dtype=tf.int32) # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
                vals = tf.math.log(top_predicted.values) * tf.cast(mask, dtype=tf.float32) # pylint: disable=no-value-for-parameter, unexpected-keyword-arg

                # collect batch states and masks
                batch_states = []
                mm = []
                for i in range(dec_in.shape[0]):
                    batch_states.append((hatt[i], h1[i], c1[i], h2[i], c2[i]))
                    mm.append(mask[i])
                initial_states.append(batch_states)
                masks.append(mm)

                for top in range(ixs.shape[1]):
                    scores = scores.write(siix, vals[:, top] + nlogsum)
                    indices = indices.write(siix, ixs[:, top])
                    prdcsrs = prdcsrs.write(siix, tf.ones(shape=ixs[:, top].shape, dtype=tf.int32) * h)
                    siix += 1
            scores = tf.transpose(scores.stack(), [1, 0])
            indices = tf.transpose(indices.stack(), [1, 0])
            prdcsrs = tf.transpose(prdcsrs.stack(), [1, 0])

            top_k_scores = tf.math.top_k(scores, k=actual_beam_size)
            top_k_indices = tf.gather(indices, top_k_scores.indices, batch_dims=-1) # pylint: disable=no-value-for-parameter
            out_sentence = out_sentence.write(t, top_k_indices)
            top_k_prdcsrs = tf.gather(prdcsrs, top_k_scores.indices, batch_dims=-1) # pylint: disable=no-value-for-parameter
            out_predecessors = out_predecessors.write(t, top_k_prdcsrs)
            for beam_ix in range(actual_beam_size):
                next_in = tf.expand_dims(top_k_indices[:, beam_ix], -1)
                for batch_ix in range(summaries.shape[0]):
                    if batch_ix == 0:
                        hatt, h1, c1, h2, c2 = initial_states[top_k_prdcsrs[batch_ix, beam_ix].numpy()][batch_ix]
                        f = lambda x: tf.expand_dims(x, 0)
                        hatt, h1, c1, h2, c2 = f(hatt), f(h1), f(c1), f(h2), f(c2)
                        mask = f(masks[top_k_prdcsrs[batch_ix, beam_ix].numpy()][batch_ix])
                    else:
                        hatt2, h12, c12, h22, c22 = initial_states[top_k_prdcsrs[batch_ix, beam_ix].numpy()][batch_ix]
                        f = lambda x, y: tf.concat([x, tf.expand_dims(y, 0)], axis=0) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                        hatt, h1, c1, h2, c2 = f(hatt, hatt2), f(h1, h12), f(c1, c12), f(h2, h22), f(c2, c22)
                        mask = f(mask, masks[top_k_prdcsrs[batch_ix, beam_ix].numpy()][batch_ix])
                i_s = (hatt, h1, c1, h2, c2)
                hypotheses.append((next_in, mask, top_k_scores.values[:, beam_ix], i_s))

        out_predecessors = out_predecessors.stack()
        out_sentence = out_sentence.stack()

        out_sentence = tf.transpose(out_sentence, [1, 0, 2])
        out_predecessors = tf.transpose(out_predecessors, [1, 0, 2])

        return out_sentence, out_predecessors
