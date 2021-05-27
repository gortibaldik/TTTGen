import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    """Embedding layer for sequences of records (4-tuples type, entity, value, home/away flag)"""
    def __init__( self
                , ent_vocab_size
                , ent_emb_dim
                , val_vocab_size
                , val_emb_dim
                , tp_vocab_size
                , tp_emb_dim
                , ha_vocab_size
                , ha_emb_dim):
        """ Initialize the internal embedding layers

        Args:
            ent_vocab_size  (int): size of the vocabulary of entities
            ent_emb_dim     (int): embedding dimensionality to which project the entity part of input record
            val_vocab_size  (int): size of the vocabulary of values
            val_emb_dim     (int): embedding dimensionality to which project the value part of input record
            tp_vocab_size   (int): size of the vocabulary of types
            tp_emb_dim      (int): embedding dimensionality to which project the type part of input record
            ha_vocab_size   (int): size of the vocabulary of home/away flags
            ha_emb_dim      (int): embedding dimensionality to which project the home/away flag part of input record
        """
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
        """ inputs should be organized as (types, entities, values, home/away flags)"""
        tp, ent, val, ha = inputs
        tp_emb = self.tp_emb(tp)
        ent_emb = self.ent_emb(ent)
        val_emb = self.val_emb(val)
        ha_emb = self.ha_emb(ha)
        return tp_emb, ent_emb, val_emb, ha_emb

class MLPEncodingCell(tf.keras.layers.Layer):
    """ MLPEncodingCell is a subclass of tf.keras.layers.Layer which aims to be used as the Cell argument in the tf.keras.layers.RNN interface

    The tf.keras.layers.RNN interface expands the computational graph to allow sharing one MLP for multiple timesteps.
    """
    def __init__( self
                , hidden_size
                , input_size):
        """ Initialize the MLP Cell
        
        
        Args:
            hidden_size (int): dimensionality of the hidden states
            input_size  (int): dimensionality of the inputs
        """
        super(MLPEncodingCell, self).__init__()
        self._MLP = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.state_size = tf.TensorShape([self.hidden_size])
    
    def call( self
            , inputs
            , states):
        # states are unused as no hidden state is shared between
        # different time-steps
        outputs = self._MLP(inputs)
        return outputs, outputs

class ContentSelectionCell(tf.keras.layers.Layer):
  """ ContentSelectionCell is a subclass of tf.keras.layers.Layer which should be used as the Cell argument in tf.keras.layers.RNN interface

  ContentSelectionCell traverse all the time-steps and compute self-attention to provide gating mechanism and regulate
  flow of the information from the record (explanation can be found in section 4.4.1 of my thesis)
  """
  def __init__( self
              , attention_type
              , max_timesteps
              , hidden_size
              , input_size):
    """ Initialize the ContentSelectionCell
        
        
        Args:
            attention_type  (callable):  method which initializes the attention mechanism used as the self attention
            max_timesteps   (int):       the lenght of the longest sequence to be processed
            hidden_size     (int):      dimensionality of the hidden states
            input_size      (int):      dimensionality of the inputs
    """
    super(ContentSelectionCell, self).__init__()
    self._attention = attention_type()
    self._non_linearity = tf.keras.layers.Dense(hidden_size, activation='sigmoid')
    self._last_alignment = None
    self.hidden_size= hidden_size
    self.input_size = input_size
    self.state_size = [tf.TensorShape((max_timesteps, input_size)), tf.TensorShape(())]

  def call(self, x, states, training=False):
    """

    Args:
        x  (tensor):                                                    the input at the actual time-step
        states (2-tuple [all_the_timesteps, actual_timestep_number]):   all_the_timesteps is used for computation of the self attention
            actual_timestep_number serves to mask out the actual timestep from all_the_timesteps during computation of attention
    """
    enc_outs, actual_step = states
    context, self._last_alignment = self._attention( x
                                                   , enc_outs
                                                   , mask_step=actual_step)
    actual_step += 1
    att = self._non_linearity(tf.concat([x, context], axis=-1)) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    content_selected = att * x
    return content_selected, (enc_outs, actual_step)

class ContentPlanDecoderCell(tf.keras.layers.Layer):
    """ ContentPlanDecoderCell, in thesis called ContentPlanningDecoder, is a pointer network which returns attention alignment as its output
    
    Further explanation of pointer networks can be found in section 3.3 of the thesis
    """
    def __init__( self
                , hidden_size
                , attention_type
                , batch_size
                , dropout_rate=0):
        """ Initialize the ContentSelectionCell
            
            
            Args:
                hidden_size     (int):      dimensionality of the hidden states
                attention_type  (callable): method which initializes the attention mechanism
                batch_size      (int):      size of batch
                dropout_rate    (float):    rate at which to drop cells and corresponding connections at
                                            the outputs of the internal LSTM layer
        """
        
        super(ContentPlanDecoderCell, self).__init__()
        self._rnn = tf.keras.layers.LSTMCell( hidden_size
                                            , recurrent_initializer='glorot_uniform'
                                            , dropout=dropout_rate)
        self._attention = attention_type(return_non_softmax_alignment=True)
        self._batch_size = batch_size
        self.state_size = [ tf.TensorShape([hidden_size])
                          , tf.TensorShape([hidden_size])]

    def call( self, x, states, training=False):
        """
        Args:
            x       (2-tuple [actual_timestep, all_timesteps]): actual_timestep is used as the input ot the internal LSTM layers,
                all_timesteps for attention computation
            states (2-tuple [h, c]):                           hidden states of the internal LSTM layer
        """
        x, enc_outs = x
        h, c = states
        output, (h, c) = self._rnn(x, (h, c), training=training)
        context, alignment = self._attention( output
                                            , enc_outs)
        return (context, alignment), (h, c)

class DotAttention(tf.keras.layers.Layer):
    """ Luong style DotAttention """

    def __init__( self, return_non_softmax_alignment=False):
        """ Initialize the DotAttention layer

        Args:
            return_non_softmax_alignment (bool): whether to return softmax of the score or the score itself
        -----
        score is explained in section 3.2 of the thesis
        """
        super(DotAttention, self).__init__()
        self._return_non_softmax_alignment = return_non_softmax_alignment

    def call( self
            , actual_hidden
            , all_encs
            , mask_step : int = None):
        """
        Args:
            actual_hidden (tensor): the actual output of the decoder
            all_encs      (tensor): all the outputs of the encoder
            mask_step     (int):    if None no time-step in the encoder is masked, otherwise mask_step-th timestep is masked
        """
        actual_hidden = tf.expand_dims(actual_hidden, 1)
        score = tf.matmul(actual_hidden, all_encs, transpose_b=True)
        score = tf.squeeze(score, [1])

        # create mask
        if mask_step is not None:
            mask = tf.one_hot(tf.ones(all_encs.shape[0], dtype=tf.int32) * mask_step
                             , all_encs.shape[1]
                             , on_value=0.0
                             , off_value=1.0)
        else:
            mask = tf.ones(score.shape, dtype=tf.float32)
        # for softmax to not consider value at all, tf.float32.min must be
        # substitued, then the output of softmax would sum up to 1 and
        # output at the place of masked value would be 0
        score = tf.where(mask==0.0, tf.float32.min, score)
        alignment = tf.nn.softmax(score)
        context = tf.reduce_sum(tf.expand_dims(alignment, -1) * all_encs, axis=1)
        if self._return_non_softmax_alignment:
            return context, score
        else:
            return context, alignment

class ConcatAttention(tf.keras.layers.Layer):
    """ Bahdanau style concat attention """
    def __init__(self, units, return_non_softmax_alignment=False):
        """ Initialize the DotAttention layer

        Args:
            units                        (int):  the hidden dimensionality to which the inputs are transformed
            return_non_softmax_alignment (bool): whether to return softmax of the score or the score itself
        -----
        score is explained in section 3.2 of the thesis
        """
        super(ConcatAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self._return_non_softmax_alignment = return_non_softmax_alignment

    def call( self
            , actual_hidden
            , all_encs
            , mask_step=None):
        """
        Args:
            actual_hidden (tensor): the actual output of the decoder
            all_encs      (tensor): all the outputs of the encoder
            mask_step     (int):    if None no time-step in the encoder is masked, otherwise mask_step-th timestep is masked
        """
        actual_hidden = tf.expand_dims(actual_hidden, 1)
        score = self.V(tf.nn.tanh(
                      self.W1(actual_hidden) + self.W2(all_encs)))
        score = tf.squeeze(score, [2])
        # create mask
        if mask_step is not None:
            mask = tf.one_hot(tf.ones(all_encs.shape[0], dtype=tf.int32) * mask_step
                             , all_encs.shape[1]
                             , on_value=0.0
                             , off_value=1.0)
        else:
            mask = tf.ones(all_encs.shape, dtype=tf.float32)
        # for softmax to not consider value at all, tf.float32.min must be
        # substitued, then the output of softmax would sum up to 1 and
        # output at the place of masked value would be 0
        score = tf.where(mask==0.0, tf.float32.min, score)
        alignment = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(tf.expand_dims(alignment, -1) * all_encs, axis=1)
        if self._return_non_softmax_alignment:
            return context, score
        else:
            return context, alignment

class DecoderRNNCell(tf.keras.layers.Layer):
    """ The DecoderRNNCell should be used as the Decoder part of the EncoderDecoderBasic model and EncoderDecoderContentSelection model

    This is the baseline decoder as explained in section 4.2.2
    """
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , decoder_rnn_dim
                , batch_size
                , attention
                , dropout_rate=0):
        """ Initialize the DecoderRNNCell

        Args:
            word_vocab_size  (int):      size of the vocabulary of words in the summaries
            word_emb_dim     (int):      embedding dimensionality to which project the word from the summary
            decoder_rnn_dim  (int):      the hidden dimensionality to which the inputs are transformed
            batch_size       (int):      size of batch
            attention        (callable): method which initializes the attention mechanism
            dropout_rate     (float):    rate at which to drop cells and corresponding connections at
                                                     the outputs of the internal LSTM layer
        """
        super(DecoderRNNCell, self).__init__()
        self._word_emb_dim = word_emb_dim
        self._word_vocab_size = word_vocab_size
        self._embedding = tf.keras.layers.Embedding(word_vocab_size, word_emb_dim)
        self._rnn_1 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform'
                                              , dropout=dropout_rate)
        self._rnn_2 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform'
                                              , dropout=dropout_rate)
        self._fc_1 = tf.keras.layers.Dense( decoder_rnn_dim
                                          , activation='tanh')
        self._fc_2 = tf.keras.layers.Dense( word_vocab_size
                                          , activation='softmax')
        self._attention = attention()
        self._batch_size = batch_size
        self._hidden_size = decoder_rnn_dim
        self._last_alignment = None
        self.state_size = [ tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size])]

    def call(self, x, states, training=False):
        """
        Args:
            x        (2-tuple [actual_timestep, all_timesteps]): actual_timestep is used as the input ot the internal LSTM layers,
                                                                 all_timesteps for attention computation
            states   (2-tuple [last_att, h1, c1, h2, c2]):       hidden states of the internal LSTM layer and Luong style input feeding
            training (bool):                                     whether to apply dropout on the LSTM layers or not
        """
        x, enc_outs = x
        emb = self._embedding(x)
        emb = tf.squeeze(emb)
        last_hidden_attn, h1, c1, h2, c2 = states
        emb_att = tf.concat([emb, last_hidden_attn], axis=-1) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        seq_output, (h1, c1) = self._rnn_1( emb_att, (h1, c1), training=training)
        seq_output, (h2, c2) = self._rnn_2( seq_output, (h2, c2), training=training)
        context, self._last_alignment = self._attention(h2, enc_outs)
        concat_ctxt_h2 = tf.concat([context, h2], axis=-1) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        hidden_att = self._fc_1(concat_ctxt_h2)
        result = self._fc_2(hidden_att)

        return result, (hidden_att, h1, c1, h2, c2)

class DecoderRNNCellJointCopy(tf.keras.layers.Layer):
    """ The DecoderRNNCellJointCopy should be used as the Decoder part of the EncoderDecoderBasic model and EncoderDecoderContentSelection model

    This is the Joint-Copy decoder as explained in section 4.3
    """
    def __init__( self
                , word_vocab_size
                , word_emb_dim
                , decoder_rnn_dim
                , batch_size
                , attention
                , dropout_rate=0):
        """ Initialize the DecoderRNNCellJointCopy

        Args:
            word_vocab_size  (int):      size of the vocabulary of words in the summaries
            word_emb_dim     (int):      embedding dimensionality to which project the word from the summary
            decoder_rnn_dim  (int):      the hidden dimensionality to which the inputs are transformed
            batch_size       (int):      size of batch
            attention        (callable): method which initializes the attention mechanism
            dropout_rate     (float):    rate at which to drop cells and corresponding connections at
                                                     the outputs of the internal LSTM layer
        """
        super(DecoderRNNCellJointCopy, self).__init__()
        self._word_emb_dim = word_emb_dim
        self._word_vocab_size = word_vocab_size
        self._embedding = tf.keras.layers.Embedding(word_vocab_size, word_emb_dim)
        self._rnn_1 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform'
                                              , dropout=dropout_rate)
        self._rnn_2 = tf.keras.layers.LSTMCell( decoder_rnn_dim
                                              , recurrent_initializer='glorot_uniform'
                                              , dropout=dropout_rate)
        self._hidden_transform = tf.keras.layers.Dense( decoder_rnn_dim
                                          , activation='tanh')
        self._gen_prob_transform = tf.keras.layers.Dense( word_vocab_size
                                          , activation='softmax')
        self._fc_3 = tf.keras.layers.Dense( 1
                                          , activation='sigmoid')
        self._attention_copy = attention()
        self._attention_generate = attention()
        self._batch_size = batch_size
        self._hidden_size = decoder_rnn_dim
        self._last_copy_prob = None
        self._last_gen_prob = None
        self._last_switch = None
        self._i = 0
        self.state_size = [ tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size]),
                            tf.TensorShape([self._hidden_size])]
    
    def call(self, x, states, training=False):
        """
        Args:
            x        (2-tuple [actual_timestep, all_timesteps]): actual_timestep is used as the input ot the internal LSTM layers,
                                                                 all_timesteps for attention computation
            states   (2-tuple [last_att, h1, c1, h2, c2]):       hidden states of the internal LSTM layer and Luong style input feeding
            training (bool):                                     whether to apply dropout on the LSTM layers or not
        """
        x, enc_outs, enc_ins = x
        last_hidden_attn, h1, c1, h2, c2 = states
        
        # embedding        
        emb = self._embedding(x)
        emb = tf.squeeze(emb)
        emb_att = tf.concat([emb, last_hidden_attn], axis=-1) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        # 2-layer LSTM decoder
        seq_output, (h1, c1) = self._rnn_1( emb_att, (h1, c1), training=training)
        seq_output, (h2, c2) = self._rnn_2( seq_output, (h2, c2), training=training)
        
        # attention for generation
        context, _ = self._attention_generate(h2, enc_outs)
        concat_ctxt_h2 = tf.concat([context, h2], axis=-1) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        hidden_att = self._hidden_transform(concat_ctxt_h2)
        self._last_gen_prob = self._gen_prob_transform(hidden_att)
        self._last_switch = self._fc_3(hidden_att)
    
        # copy probabilities
        _, alignment = self._attention_copy(h2, enc_outs)

        weighted_ins = enc_ins * tf.expand_dims(alignment, -1)
        self._last_copy_prob = tf.reduce_sum(weighted_ins, axis=1)

        result = self._last_gen_prob * (tf.ones(shape=[self._batch_size, 1]) - self._last_switch) + \
                    self._last_copy_prob * self._last_switch

        return result, (hidden_att, h1, c1, h2, c2)
