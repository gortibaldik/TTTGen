import tensorflow as tf
import numpy as np
import sys

from neural_nets.layers import DecoderRNNCellJointCopy

class EncoderDecoderBasic(tf.keras.Model):
    """ EncoderDecoder model which allows usage of different Encoders (Encoder, EncoderCS, EncoderCSBi) and Decoders(DecoderRNNCell, DecoderRNNCellJointCopy)

    Encoder should encode the input records to a representation out of which the decoder would generate (decode) the output text
    """
    def __init__( self
                , encoder
                , decoder_cell):
        """ Initialize EncoderDecoderBasic
        Args:
            encoder:        one of Encoder, EncoderCS, EncoderCSBi
            decoder_cell:   one of DecoderRNNCell, DecoderRNNCellJointCopy
        """
        super(EncoderDecoderBasic, self).__init__()
        self._encoder = encoder
        self._decoder_cell = decoder_cell

    def compile( self
               , optimizer
               , loss_fn
               , scheduled_sampling_rate
               , truncation_size
               , truncation_skip_step):
        """ Prepare the model for training, evaluation and prediction

        Assigns optimizers, losses, initiates training hyperparameters, sets up eager execution,
        which enables us to use different settings for training (we use graph execution during training)
        and evaluation and prediction (where we use eager execution)

            Args:
            optimizer               (optimizer):    optimizer used to minimize the txt loss
            loss_fn                 (loss):         loss function
            scheduled_sampling_rate (float):        frequency at which the gold outputs from the previous time-steps are fed into the network
                                                    (number between 0 and 1, 1 means regular training)
            truncation_size         (int):          t_2 argument of TBPTT (explained in section 4.1 of the thesis)
            truncation_skip_step    (int):          t_1 argument of TBPTT (should be lower than or equal to t_2)
        """
        super(EncoderDecoderBasic, self).compile(run_eagerly=True)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._train_metrics = [ tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
                              , tf.keras.metrics.SparseCategoricalCrossentropy(name='loss')]
        self._val_metrics = { "val_accuracy" : tf.keras.metrics.Accuracy(name='accuracy')
                            , "val_loss" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss')}
        self._scheduled_sampling_rate = scheduled_sampling_rate
        self._truncation_skip_step = truncation_skip_step
        self._truncation_size = truncation_size
        self._generator = tf.random.Generator.from_non_deterministic_state()

    def _calc_loss( self, x, y):
        """ use only the non-pad values to calculate the loss"""
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss_ = self._loss_fn(y, x)
        mask = tf.cast(mask, loss_.dtype)
        loss_ *= mask
        for metric in self._train_metrics:
            metric.update_state(y, x, sample_weight=mask)
        return tf.reduce_mean(loss_)

    @tf.function
    def bppt_step( self
                 , batch_data
                 , last_out
                 , initial_state=None):
        """ performs one forward and backward pass

        the first pass with given arguments creates a computational graph which 
        is used in the other passes

        Args:
            batch_data:     inputs and targets for the text decoder, input tables, and
                            generated probabilities for scheduled sampling
                            (dec_in, dec_targets, gen_or_teach, *tables)
            last_out:       last output of the decoder
            initial_state:  initial state for the decoder
        
        Returns:
            self._truncation_skip_step-th hidden state
            argmax of self._truncation_skip_step-th prediction of the network
        """

        # debugging outputs - tf.function calls python functions only 
        # during tracing when a computation graph is created
        # we report how many times the function is traced and with which arguments
        print("tracing tf.function with args:", file=sys.stderr)
        print(f"len(batch_data) : {len(batch_data)}", file=sys.stderr)
        for ix, d in enumerate(batch_data):
            print(f"batch_data[{ix}].shape : {d.shape}", file=sys.stderr)
        print(f"last_out.shape : {last_out.shape}", file=sys.stderr)
        print(f"last_out.dtype : {last_out.dtype}", file=sys.stderr)
        if initial_state is None:
          print(f"initial_state is None", file=sys.stderr)
        else:
          for ix, d in enumerate(initial_state):
            print(f"initial_state[{ix}].shape : {d.shape}", file=sys.stderr)
        
        loss = 0
        dec_in, targets, gen_or_teach, *tables = batch_data
        final_state = None
        final_last_out = None
        with tf.GradientTape() as tape:
            enc_outs, *last_hidden_rnn = self._encoder(tables, training=True)
            if initial_state is None:
                initial_state = [ last_hidden_rnn[-1]
                                , *last_hidden_rnn ]
            # prepare states and inputs for the decoder
            if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
                print("using joint copy mechanism !")
                enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
            else:
                print("using vanilla attention")
                aux_inputs = (enc_outs,)
            print(f"---\n", file=sys.stderr)

            states = initial_state
            for t in range(dec_in.shape[1]):
                if (self._truncation_skip_step is not None) and (t == self._truncation_skip_step):
                    final_state = states
                    final_last_out = last_out
                if gen_or_teach[t] > self._scheduled_sampling_rate:
                    _input = last_out
                else:
                    _input = dec_in[:, t, :]
                last_out, states = self._decoder_cell( (_input, *aux_inputs)
                                                     , states=states
                                                     , training=True)
                # calculate loss and collect metrics
                loss += self._calc_loss( last_out
                                       , targets[:, t])
                # prepare new input for the decoder (used with (1 - scheduled_sampling_rate) probability)
                last_out = tf.expand_dims(tf.cast(tf.argmax(last_out, axis=1), tf.int16), -1)

        variables = []
        # linear transformation layer is trained only when the initial_state is None
        # when it prepares the initial states for the decoder
        for var in self._encoder.trainable_variables + self._decoder_cell.trainable_variables:
            if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0'):
                variables.append(var)

        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))

        if (self._truncation_skip_step is None) or (self._truncation_skip_step == dec_in.shape[1]):
            final_state = states
            final_last_out = last_out

        return final_state, final_last_out


    @property
    def metrics(self):
        """returns the list of metrics that should be reset at the start and end of training epoch and evaluation"""
        return self._train_metrics + list(self._val_metrics.values())


    def train_step(self, batch_data):
        """ perform one train_step during model.fit

        Args:
            batch_data: data to train on in format (summaries, *tables)
        """
        summaries, *tables = batch_data

        # prepare inputs for the decoder
        sums = tf.expand_dims(summaries, axis=-1)

        # scheduled sampling may force the text decoder to generate
        # from its last prediction even at the first step
        # by setting last_out = sums[:, 0] we erase differences between
        # scheduled sampling and teacher forcing at the first timestep
        last_out = sums[:, 0]
        start = 0
        length = summaries.shape[1]
        state = None
        for end in range(self._truncation_size, length-1, self._truncation_skip_step):
            # gen_or_teach is the [0,1] vector which contains value for each time-step
            # if the value for the timestep is higher than self.scheduled_sampling_rate
            # the text decoder is forced to generate from its last prediction
            gen_or_teach = np.zeros(shape=(end-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)

            # prepare data for teacher forcing, scheduled sampling etc.
            truncated_data = ( sums[:, start:end, :]
                             , summaries[:, start+1:end+1]
                             , tf.convert_to_tensor(gen_or_teach)
                             , *tables)
            
            # run the backpropagation on truncated sequence
            state, last_out =  self.bppt_step( truncated_data
                                             , last_out
                                             , initial_state=state)
            start += self._truncation_skip_step
        # finish the truncated bppt if the truncation_size cannot divide properly
        # the length of sequence
        if (length - self._truncation_size) % self._truncation_skip_step != 0:
            # gen_or_teach is the [0,1] vector which contains value for each time-step
            # if the value for the timestep is higher than self.scheduled_sampling_rate
            # the text decoder is forced to generate from its last prediction
            gen_or_teach = np.zeros(shape=(length-1-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)

            # prepare data for teacher forcing, scheduled sampling etc.
            truncated_data = ( sums[:, start:length-1, :]
                                , summaries[:, start+1:length]
                                , tf.convert_to_tensor(gen_or_teach)
                                , *tables)

            # run the backpropagation on truncated sequence
            state, last_out = self.bppt_step( truncated_data
                                            , last_out
                                            , initial_state=state)
        return dict([(metric.name, metric.result()) for metric in self._train_metrics])


    def test_step(self, batch_data):
        """ perform one test_step during model.evaluate

        Args:
            batch_data: data to train on in format (summaries, *tables)
        """
        summaries, *tables = batch_data
        # prepare summaries
        max_sum_size = summaries.shape[1] - 1
        dec_inputs = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]
        targets = summaries[:, 1:max_sum_size+1]

        enc_outs, *last_hidden_rnn = self._encoder(tables, training=False)

        # prepare states and inputs for the decoder
        if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (enc_outs,)

        initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]
        dec_in = dec_inputs[:, 0, :] # start tokens

        result_preds = np.zeros(targets.shape, dtype=np.int)

        # decode text
        for t in range(targets.shape[1]):
            pred, initial_state = self._decoder_cell( (dec_in, *aux_inputs)
                                                    , initial_state
                                                    , training=False)

            mask = tf.math.logical_not(tf.math.equal(targets[:, t], 0))
            self._val_metrics['val_loss'].update_state( targets[:, t]
                                                      , pred
                                                      , sample_weight=mask )

            predicted_ids = tf.argmax(pred, axis=1).numpy()
            result_preds[:, t] = predicted_ids
            dec_in = tf.expand_dims(targets[:, t], axis=1)

        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        self._val_metrics['val_accuracy'].update_state( targets
                                                      , result_preds
                                                      , sample_weight=mask)

        return dict([(metric.name, metric.result()) for metric in self._val_metrics.values()])

    def predict_step(self, data):
        """ perform one predict_step during model.predict

        Args:
            batch_data: data to train on in format (summaries, *tables)
        """
        summaries, *tables = data

        # retrieve start tokens
        dec_inputs = tf.expand_dims(summaries, axis=-1)
        dec_in = dec_inputs[:, 0, :] # start tokens

        enc_outs, *last_hidden_rnn = self._encoder(tables, training=False)

        if isinstance(self._decoder_cell, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), self._decoder_cell._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (enc_outs,)

        initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]

        result_preds = np.zeros(summaries.shape, dtype=np.int)

        # greedy decoding
        for t in range(summaries.shape[1]):
            pred, initial_state = self._decoder_cell( (dec_in, *aux_inputs)
                                                    , initial_state
                                                    , training=False)

            predicted_ids = tf.argmax(pred, axis=1).numpy()
            result_preds[:, t] = predicted_ids
            dec_in = tf.expand_dims(predicted_ids, axis=1)

        return result_preds