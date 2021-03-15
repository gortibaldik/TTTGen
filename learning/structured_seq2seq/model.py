import tensorflow as tf

"""
The model is a classical encoder-decoder architecture
with field-gating lstm unit and dual attention units
implementing the paper: https://arxiv.org/pdf/1711.09724.pdf
taking great amount of inspiration in the released original code: https://github.com/tyliupku/wiki2bio/
"""


class Embedding(tf.keras.layers.Layer):
    def __init__( self
                , word_vocab_size
                , word_embedding_dim
                , field_vocab_size
                , field_embedding_dim
                , position_vocab_size
                , position_embedding_dim):
        super(Embedding, self).__init__()
        self.word_embedding = tf.keras.layers.Embedding(word_vocab_size, word_embedding_dim)
        self.field_embedding = tf.keras.layers.Embedding(field_vocab_size, field_embedding_dim)
        self.pos_embedding = tf.keras.layers.Embedding(position_vocab_size, position_embedding_dim)
        self.rev_pos_embedding = tf.keras.layers.Embedding(position_vocab_size, position_embedding_dim)

    def call(self, inputs, **kwargs):
        """
        Based on paper, the operation done in embeddings:
        embed all the arguments, then concatenate fields pos and rpos to f_embed
        :return word_embeddings and concatenated field_emb, pos_emb and rpos_emb
        """
        words, fields, pos, rpos = inputs
        word_embeddings = self.word_embedding(words)
        field_embeddings = self.field_embedding(fields)
        pos_embeddings = self.pos_embedding(pos)
        rpos_embeddings = self.rev_pos_embedding(rpos)
        field_pos_embeddings = tf.concat([field_embeddings, pos_embeddings, rpos_embeddings], axis=2)
        return word_embeddings, field_pos_embeddings


class FieldGatingLSTMCell(tf.keras.layers.Layer):
    def __init__( self
                , hidden_size
                , input_size
                , field_size):
        """
        :param hidden_size: number of FieldGatingLSTMCells
        :param input_size: words embedding dimension
        :param field_size: fields embedding dimension
        """
        super(FieldGatingLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.state_size = [tf.TensorShape([self.hidden_size]), tf.TensorShape([self.hidden_size])]

        self.basicLSTM = tf.keras.layers.Dense(
            input_shape=[None, self.input_size+self.hidden_size],
            units=4*self.hidden_size,
            activation='linear'
        )
        self.fieldGating = tf.keras.layers.Dense(
            input_shape=[None, self.field_size],
            units= 2*self.hidden_size,
            activation='linear'
        )

    def call( self
            , inputs
            , states):
        word, field = inputs
        h_old, c_old = states
        x = tf.concat([word, h_old], 1)

        i, f, o, c = tf.split(self.basicLSTM(x), 4, 1)
        l, z = tf.split(self.fieldGating(field), 2, 1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f+1.0) # need to investigate, why "+1.0" is there
        o = tf.sigmoid(o)
        c = tf.tanh(c)

        l = tf.sigmoid(l)
        z = tf.tanh(z)

        c = f * c_old + i * c + l * z
        h = o * tf.tanh(c)
        out, state = h, (h, c)

        return out, state
