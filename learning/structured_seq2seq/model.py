import tensorflow as tf
from layers import Embedding, FieldGatingLSTMCell, DualAttention


class Encoder(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , field_vocab_size
                , field_emb_dim
                , pos_vocab_size
                , pos_emb_dim
                , fglstm_dim):
        super(Encoder, self).__init__()
        self.embedding_layer = Embedding( word_vocab_size
                                        , word_emb_dim
                                        , field_vocab_size
                                        , field_emb_dim
                                        , pos_vocab_size
                                        , pos_emb_dim)
        self._field_pos_emb_dim = self.embedding_layer.get_output_shape()[1][2]
        self.cell = FieldGatingLSTMCell( fglstm_dim
                                       , word_emb_dim
                                       , self._field_pos_emb_dim)
        self.rnn = tf.keras.layers.RNN( self.cell
                                      , return_sequences=True
                                      , return_state=True)

    def get_field_pos_emb_dim(self):
        return self._field_pos_emb_dim

    def call(self, inputs):
        table_embeddings, field_pos_embeddings = self.embedding_layer(inputs)
        outputs, h, c = self.rnn((table_embeddings, field_pos_embeddings))
        return outputs, (h, c), field_pos_embeddings


class Decoder(tf.keras.Model):
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , field_pos_emb_dim
                , lstm_dim):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(word_vocab_size, word_emb_dim)
        self.rnn = tf.keras.layers.LSTM( lstm_dim
                                       , return_state=True
                                       , recurrent_initializer='glorot_uniform')
        self.attention = DualAttention(lstm_dim, lstm_dim, field_pos_emb_dim)
        self.fc = tf.keras.layers.Dense(
            units=word_vocab_size,
            activation='softmax'
        )
        self.hidden_state = None
        self.carry_state = None

    def initialize_batch( self
                        , encoder_outputs
                        , field_embeddings
                        , hidden_state
                        , carry_state):
        """
        This method should be called at the start of each batch,
        it initializes hidden state of the decoder for all the timesteps
        of the batch and calculate time-invariant values of the attention
        """
        self.hidden_state = hidden_state
        self.carry_state = carry_state
        self.attention.calc_timestep_consts(encoder_outputs, field_embeddings)

    def call( self
            , input):
        emb = self.embedding(input)
        print(f"emb.shape : {emb.shape}")
        seq_output, self.hidden_state, self.carry_state = self.rnn(
            emb,
            initial_state=(self.hidden_state, self.carry_state)
        )
        attention_vector, attention_weights = self.attention(self.hidden_state)
        output = self.fc(attention_vector)
        return output, attention_weights
