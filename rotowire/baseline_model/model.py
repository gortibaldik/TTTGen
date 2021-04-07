import tensorflow as tf
from baseline_model.layers import Embedding, MLPEncodingCell, DotAttention


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
        self._linear_transform = tf.keras.layers.Dense(4*hidden_size, use_bias=False)
        self._batch_size = batch_size

    def change_batch_size(self, new_batch_size : int):
        self._batch_size = new_batch_size
    
    def call(self, inputs):
        embedded = self._embedding(inputs)
        embedded = tf.concat(embedded, axis=2)
        all_states, _ = self._rnn(embedded)
        pooled = self._pooling(all_states)
        pooled = tf.reshape(pooled, [self._batch_size, -1])
        trans = self._linear_transform(pooled)
        hidden_1, cell_1, hidden_2, cell_2 = tf.split(trans, 4 , 1)
        return all_states, hidden_1, cell_1, hidden_2, cell_2


class Decoder(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , decoder_rnn_dim
                , batch_size):
        super(Decoder, self).__init__()
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
        self._attention = DotAttention()
        self._batch_size = batch_size
    
    def call(self, x, last_hidden_rnn, last_hidden_att, encoder_outputs):
        emb = self._embedding(x)
        emb_att = tf.concat([emb, tf.expand_dims(last_hidden_att, 1)], axis=-1)
        seq_output, h1, c1 = self._rnn_1( emb_att
                                        , initial_state=(last_hidden_rnn[0], last_hidden_rnn[1]))
        seq_output, h2, c2 = self._rnn_2( seq_output
                                        , initial_state=(last_hidden_rnn[2], last_hidden_rnn[3]))
        context, alignment = self._attention(h2, encoder_outputs)
        concat_ctxt_h2 = tf.concat([context, h2], axis=-1)
        hidden_att = self._fc_1(concat_ctxt_h2)
        result = self._fc_2(hidden_att)

        return result, hidden_att, alignment, h1, c1, h2, c2
