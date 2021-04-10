import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    def __init__( self
                , ent_vocab_size
                , ent_emb_dim
                , val_vocab_size
                , val_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim):
        super(Embedding, self).__init__()
        # define embedding layers
        self.ent_emb = tf.keras.layers.Embedding(ent_vocab_size, ent_emb_dim)
        self.val_emb = tf.keras.layers.Embedding(val_vocab_size, val_emb_dim)
        self.tp_emb = tf.keras.layers.Embedding(tp_vocab_size, tp_emb_dim)
        self.ha_emb = tf.keras.layers.Embedding(ha_vocab_size, ha_emb_dim)
        self._os = (
            tf.TensorShape([None, None, tp_emb_dim]),
            tf.TensorShape([None, None, ent_emb_dim]),
            tf.TensorShape([None, None, val_emb_dim]),
            tf.TensorShape([None, None, ha_emb_dim])
        )

    def get_output_shape(self):
        return self._os

    def call(self, inputs, **kwargs):
        tp, ent, val, ha = inputs
        tp_emb = self.tp_emb(tp)
        ent_emb = self.ent_emb(ent)
        val_emb = self.val_emb(val)
        ha_emb = self.ha_emb(ha)
        return tp_emb, ent_emb, val_emb, ha_emb

class MLPEncodingCell(tf.keras.layers.Layer):
    def __init__( self
                , hidden_size
                , input_size):
        """
        :param hidden_size: number of MLPEncodingCells
        :param input_size: embedding dimension
        """
        super(MLPEncodingCell, self).__init__()
        self._MLP = tf.keras.layers.Dense(hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.state_size = tf.TensorShape([self.hidden_size])
    
    def call( self
            , inputs
            , states):
        outputs = self._MLP(inputs)
        return outputs, outputs

class DotAttention(tf.keras.layers.Layer):
    def __init__( self):
        super(DotAttention, self).__init__()
    
    def call( self
            , actual_hidden
            , all_encs):
        actual_hidden = tf.expand_dims(actual_hidden, 1)
        score = tf.matmul(actual_hidden, all_encs, transpose_b=True)
        score = tf.squeeze(score, [1])
        alignment = tf.nn.softmax(score)
        context = tf.reduce_sum(tf.expand_dims(alignment, -1) * all_encs, axis=1)
        return context, alignment

class DecoderRNNCell(tf.keras.layers.Layer):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , decoder_rnn_dim
                , batch_size):
        super(DecoderRNNCell, self).__init__()
        self._word_emb_dim = word_emb_dim
        self._embedding = tf.keras.layers.Embedding(word_vocab_size, word_emb_dim)
        self._rnn_1 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform')
        self._rnn_2 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform')
        self._fc_1 = tf.keras.layers.Dense( decoder_rnn_dim
                                          , activation='tanh')
        self._fc_2 = tf.keras.layers.Dense( word_vocab_size
                                          , activation='softmax')
        self._attention = DotAttention()
        self._batch_size = batch_size
        self._hidden_size = decoder_rnn_dim
        self._enc_outs = None
        self.state_size = [ tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size])]

    def initialize_enc_outs(self, enc_outs):
        self._enc_outs = enc_outs
    
    def get_last_alignment(self):
        return self._last_alignment

    def call(self, x, states):
        emb = self._embedding(x)
        emb = tf.squeeze(emb)
        last_hidden_attn, h1, c1, h2, c2 = states
        emb_att = tf.concat([emb, last_hidden_attn], axis=-1)
        seq_output, (h1, c1) = self._rnn_1( emb_att, (h1, c1))
        seq_output, (h2, c2) = self._rnn_2( seq_output, (h2, c2))
        context, self._last_alignment = self._attention(h2, self._enc_outs)
        concat_ctxt_h2 = tf.concat([context, h2], axis=-1)
        hidden_att = self._fc_1(concat_ctxt_h2)
        result = self._fc_2(hidden_att)

        return result, (hidden_att, h1, c1, h2, c2)