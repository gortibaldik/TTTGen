import tensorflow as tf
from neural_nets.layers import Embedding, MLPEncodingCell, ContentSelectionCell

class Encoder(tf.keras.Model):
    """ Baseline Encoder as described in section 4.2.1 of the thesis
    """
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
        """ Initialize Encoder

        Args:
            word_vocab_size         (int):      size of the vocabulary of words
            word_emb_dim            (int):      embedding dimensionality to which project the words from the summary and value and entity
                                                part of the input records
            tp_vocab_size           (int):      size of the vocabulary of types
            tp_emb_dim              (int):      embedding dimensionality to which project the type part of input record
            ha_vocab_size           (int):      size of the vocabulary of home/away flags
            ha_emb_dim              (int):      embedding dimensionality to which project the home/away flag part of input record
            entity_span             (int):      number of records belonging to one player entity
            hidden_size             (int):      dimensionality of the hidden states
            batch_size              (int):      the size of batches in the dataset
        """
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
        self._linear_transform = tf.keras.layers.Dense(4*hidden_size, use_bias=False, name="linear_transform")
        self._batch_size = batch_size

    def change_batch_size(self, new_batch_size : int):
        self._batch_size = new_batch_size

    def call(self, inputs):
        """ inputs should be organized as (types, entities, values, home/away flags)"""
        # embed the records
        embedded = self._embedding(inputs)
        embedded = tf.concat(embedded, axis=2) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        # pass through mlp
        all_states, _ = self._rnn(embedded)

        # prepare states for the decoder
        pooled = self._pooling(all_states)
        pooled = tf.reshape(pooled, [self._batch_size, -1])
        trans = self._linear_transform(pooled)
        hidden_1, cell_1, hidden_2, cell_2 = tf.split(trans, 4 , 1)

        return all_states, hidden_1, cell_1, hidden_2, cell_2

class EncoderCS(tf.keras.Model):
    """ Encoder with Content Selection mechanism as described in section 4.4.1 in the thesis

    Uses self-attention to model context awareness of the representation of the records to other records.
    """
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim
                , max_seq_size
                , hidden_size
                , attention_type
                , batch_size):
        """ Initialize EncoderCS

        Args:
            word_vocab_size         (int):      size of the vocabulary of words
            word_emb_dim            (int):      embedding dimensionality to which project the words from the summary and value and entity
                                                part of the input records
            tp_vocab_size           (int):      size of the vocabulary of types
            tp_emb_dim              (int):      embedding dimensionality to which project the type part of input record
            ha_vocab_size           (int):      size of the vocabulary of home/away flags
            ha_emb_dim              (int):      embedding dimensionality to which project the home/away flag part of input record
            max_seq_size            (int):      maximal length of the records from the dataset
            attention_type          (callable): method which initializes the attention mechanism
            hidden_size             (int):      dimensionality of the hidden states
            batch_size              (int):      the size of batches in the dataset
        """
        super(EncoderCS, self).__init__()
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
        self._content_selection_cell = ContentSelectionCell(attention_type, max_seq_size, hidden_size, hidden_size)
        self._rnn = tf.keras.layers.RNN(self._rnncell, return_sequences=True, return_state=True)
        self._cs_rnn = tf.keras.layers.RNN(self._content_selection_cell, return_sequences=True, return_state=True)
        self._batch_size = batch_size

    def change_batch_size(self, new_batch_size : int):
        self._batch_size = new_batch_size

    def call(self, inputs):
        """ inputs should be organized as (types, entities, values, home/away flags)"""
        # embed the records
        embedded = self._embedding(inputs)
        embedded = tf.concat(embedded, axis=2) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        # pass through mlp
        all_states, _ = self._rnn(embedded)

        # content selection 
        content_selected, *_ = self._cs_rnn(all_states, initial_state=(all_states, tf.zeros((), dtype=tf.int32)))
        avg = tf.reduce_mean(content_selected, axis=1)
        return content_selected, avg, avg, avg, avg

class EncoderCSBi(tf.keras.Model):
    """ Encoder with Content Selection and Bidirectional LSTM layer (encoder from CopyCSBidir model) as described in section 4.4.2 in the thesis

    A wrapper around EncoderCS with additional bidirectional LSTM layer at the top of the outputs of EncoderCS
    """
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim
                , max_seq_size
                , hidden_size
                , attention_type
                , batch_size
                , dropout_rate):
        """ Initialize EncoderCSBi

        Args:
            word_vocab_size         (int):      size of the vocabulary of words
            word_emb_dim            (int):      embedding dimensionality to which project the words from the summary and value and entity
                                                part of the input records
            tp_vocab_size           (int):      size of the vocabulary of types
            tp_emb_dim              (int):      embedding dimensionality to which project the type part of input record
            ha_vocab_size           (int):      size of the vocabulary of home/away flags
            ha_emb_dim              (int):      embedding dimensionality to which project the home/away flag part of input record
            max_seq_size            (int):      maximal length of the records from the dataset
            hidden_size             (int):      dimensionality of the hidden states
            attention_type          (callable): method which initializes the attention mechanism
            batch_size              (int):      the size of batches in the dataset
            dropout_rate            (float):    rate at which to drop cells and corresponding connections at
                                                the outputs of the internal LSTM layers of the model (number
                                                between 0 and 1, 0 means no dropout)
        """
        super(EncoderCSBi, self).__init__()
        self._encoder_cs = EncoderCS( word_vocab_size
                                    , word_emb_dim
                                    , tp_vocab_size
                                    , tp_emb_dim
                                    , ha_vocab_size
                                    , ha_emb_dim
                                    , max_seq_size
                                    , hidden_size
                                    , attention_type
                                    , batch_size)
        self._bidir = tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_size
                                                                         , return_sequences=True
                                                                         , return_state=True
                                                                         , dropout=dropout_rate)
                                                   , merge_mode='sum')

    def call(self, inputs, training=False):
        encoded, avg, *_ = self._encoder_cs(inputs)
        return self._bidir(encoded, training=training)
