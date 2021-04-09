import tensorflow as tf
from .layers import DotAttention

class Decoder(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , decoder_rnn_dim
                , batch_size):
        super(Decoder, self).__init__()
        self._word_vocab_size = word_vocab_size
        self._embedding = tf.keras.layers.Embedding(word_vocab_size, word_emb_dim)
        self._rnn_1 = tf.keras.layers.LSTM( decoder_rnn_dim
                                          , return_sequences=True
                                          , return_state=True
                                          , recurrent_initializer='glorot_uniform')
        self._rnn_2 = tf.keras.layers.LSTM( decoder_rnn_dim
                                          , return_sequences=True
                                          , return_state=True
                                          , recurrent_initializer='glorot_uniform')
        self._fc_1 = tf.keras.layers.Dense( decoder_rnn_dim
                                          , activation='tanh')
        self._fc_2 = tf.keras.layers.Dense( word_vocab_size
                                          , activation='softmax')
        self._fc_3 = tf.keras.layers.Dense( 1
                                          , activation='sigmoid')
        self._attention = DotAttention()
        self._batch_size = batch_size
    
    def call(self, encoder_x, x, last_hidden_rnn, last_hidden_att, encoder_outputs):
        encoder_x = tf.one_hot(encoder_x, self._word_vocab_size)
        emb = self._embedding(x)
        emb_att = tf.concat([emb, tf.expand_dims(last_hidden_att, 1)], axis=-1)
        seq_output, h1, c1 = self._rnn_1( emb_att
                                        , initial_state=(last_hidden_rnn[0], last_hidden_rnn[1]))
        seq_output, h2, c2 = self._rnn_2( seq_output
                                        , initial_state=(last_hidden_rnn[2], last_hidden_rnn[3]))
        context, alignment = self._attention(h2, encoder_outputs)
        weighted_inputs = encoder_x * tf.expand_dims(alignment, -1)
        copy_prob = tf.reduce_sum(weighted_inputs, axis=1)
        concat_ctxt_h2 = tf.concat([context, h2], axis=-1)
        hidden_att = self._fc_1(concat_ctxt_h2)
        generate_prob = self._fc_2(hidden_att)
        switch_prob = self._fc_3(concat_txt_h2)
        result = generate_prob * (tf.ones(self._batch_size) - switch_prob) + copy_prob * switch_prob

        return result, generate_prob, switch_prob, copy_prob, hidden_att, alignment, h1, c1, h2, c2