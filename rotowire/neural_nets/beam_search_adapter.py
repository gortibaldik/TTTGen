from neural_nets.layers import DecoderRNNCellJointCopy
from neural_nets.cp_model import ContentPlanDecoder, EncoderDecoderContentSelection

import tensorflow as tf


class BeamSearchAdapter(tf.keras.Model):
    """Adapter for the EncoderDecoderContentSelection and EncoderDecoderBasic, providing Beam Search Decoding"""
    def __init__( self
                , encoder_decoder
                , beam_size
                , eos
                , max_cp_size):
        """ Initialize BeamSearchAdapter

        Args:
            encoder_decoder:    trained model, from which we steal its internals
            beam_size:          the number of hypotheses to expand on
            eos:                vocabulary index of the eos token
            max_cp_size:        maximal size of the generated content_plan
        """
        super(BeamSearchAdapter, self).__init__()
        if isinstance(encoder_decoder, EncoderDecoderContentSelection):
            self._encoder = ContentPlanDecoder(encoder_decoder, max_cp_size)
            self._decoder_cell = encoder_decoder._text_decoder
        else:
            self._encoder = encoder_decoder._encoder
            self._decoder_cell = encoder_decoder._decoder_cell
        self._beam_size = beam_size
        self._eos = eos

    def compile(self):
        """ Enable eager execution """
        super(BeamSearchAdapter, self).compile(run_eagerly=True)

    def call(self, batch_data):
        if isinstance(self._encoder, ContentPlanDecoder):
            summaries, _, *tables = batch_data
        else:
            summaries, *tables = batch_data

        # retrieve start tokens
        dec_inputs = tf.expand_dims(summaries, axis=-1)
        dec_in = dec_inputs[:, 0, :] # start tokens

        if isinstance(self._encoder, ContentPlanDecoder):
            cp_enc_outs, cp_enc_ins, last_hidden_rnn = self._encoder(tables, training=False)
            if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
                enc_ins = tf.one_hot(tf.cast(cp_enc_ins, tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                aux_inputs = (cp_enc_outs, enc_ins) # value portion of the record needs to be copied
            else:
                aux_inputs = (cp_enc_outs,)
        else:
            enc_outs, *last_hidden_rnn = self._encoder(tables, training=False)
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

            # traverse all the hypotheses
            for h in range(len(hypotheses)):
                dec_in, mask, nlogsum, initial_state = hypotheses.pop(0)
                # if the last generated token is <<EOS>> mask everything after
                # it effectively causes the generation to stop
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

                # collect scores and associated indices of the decoded tokens
                for top in range(ixs.shape[1]):
                    scores = scores.write(siix, vals[:, top] + nlogsum)
                    indices = indices.write(siix, ixs[:, top])
                    prdcsrs = prdcsrs.write(siix, tf.ones(shape=ixs[:, top].shape, dtype=tf.int32) * h)
                    siix += 1
            scores = tf.transpose(scores.stack(), [1, 0])
            indices = tf.transpose(indices.stack(), [1, 0])
            prdcsrs = tf.transpose(prdcsrs.stack(), [1, 0])

            # keep only k best scores with associated indices, states and predecessors
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

        # return sequence of indices and predecessors that enable decoding of the sequence
        return out_sentence, out_predecessors
