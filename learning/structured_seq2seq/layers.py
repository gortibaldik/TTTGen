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
        # output shape of the layer
        self.os = (tf.TensorShape([None, None, word_vocab_size]),
                   tf.TensorShape([None, None, field_embedding_dim + 2 * position_embedding_dim]))
        # define embedding layers
        self.word_embedding = tf.keras.layers.Embedding(word_vocab_size, word_embedding_dim)
        self.field_embedding = tf.keras.layers.Embedding(field_vocab_size, field_embedding_dim)
        self.pos_embedding = tf.keras.layers.Embedding(position_vocab_size, position_embedding_dim)
        self.rev_pos_embedding = tf.keras.layers.Embedding(position_vocab_size, position_embedding_dim)

    def get_output_shape(self):
        return self.os

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

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (tf.zeros(shape=[batch_size, self.hidden_size], dtype=tf.float32),
                tf.zeros(shape=[batch_size, self.hidden_size], dtype=tf.float32))

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


class DualAttention(tf.keras.layers.Layer):
    def __init__( self
                , hidden_size
                , input_size
                , field_size):
        super(DualAttention, self).__init__()
        self.field_embs_TimeIndependent = None
        self.enc_outs_TimeIndependent = None
        self.enc_outs = None
        # field embeddings Time Independent Layer
        self.field_embs_TIL = tf.keras.layers.Dense(
            input_shape=[None, field_size],
            units=hidden_size,
            activation='tanh'
        )
        # encoder outputs Time Independent Layer
        self.enc_outs_TIL = tf.keras.layers.Dense(
            input_shape=[None, input_size],
            units=hidden_size,
            activation='tanh'
        )
        self.state_fields_L = tf.keras.layers.Dense(
            input_shape=[None, input_size],
            units=hidden_size,
            activation='tanh'
        )
        self.state_enc_outs_L = tf.keras.layers.Dense(
            input_shape=[None, input_size],
            units=hidden_size,
            activation='tanh'
        )
        self.out_gate_L = tf.keras.layers.Dense(
            input_shape=[None, 2*input_size], # [hidden_state, attention_vector]
            units=hidden_size,
            activation='tanh'
        )

    def call( self, x):
        state_tanh_f = self.state_fields_L(x)
        state_tanh_x = self.state_enc_outs_L(x)

        # `*` is element-wise multiplication
        score_enc_outs = state_tanh_x * self.enc_outs_TimeIndependent
        score_fields = state_tanh_f * self.field_embs_TimeIndependent

        # nothing written in the paper about summing along the last dimension
        # although their code does it
        score_enc_outs = tf.reduce_sum(score_enc_outs, axis=2)
        alphas = tf.nn.softmax(score_enc_outs, axis=0)

        score_fields = tf.reduce_sum(score_fields, axis=2)
        betas = tf.nn.softmax(score_fields, axis=0)

        alphas_mul_betas = alphas * betas
        gammas = tf.divide(alphas_mul_betas, (1e-6 + tf.reduce_sum(alphas_mul_betas, axis=0, keepdims=True)))

        context = tf.reduce_sum(self.enc_outs * tf.expand_dims(gammas, 2), axis = 0)

        out = self.out_gate_L(tf.concat([context, x], -1))
        return out, gammas

    def calc_timestep_consts(self, encoder_outputs, field_embeddings):
        """
         During processing of single batch, there are variables which are
         constant for each time-step, initializing them beforehand means
         (hopefuly) some reductions on training time
        """
        # transforming the inputs from batch-major to time-major
        self.enc_outs = tf.transpose(encoder_outputs, [1,0,2])
        field_embeddings = tf.transpose(field_embeddings, [1,0,2])

        self.field_embs_TimeIndependent = self.field_embs_TIL(field_embeddings)
        self.enc_outs_TimeIndependent = self.enc_outs_TIL(self.enc_outs)
