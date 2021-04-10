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
