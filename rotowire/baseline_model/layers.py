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
        