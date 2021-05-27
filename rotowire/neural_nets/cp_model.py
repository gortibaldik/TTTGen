import tensorflow as tf
import numpy as np
import os
from .layers import DecoderRNNCellJointCopy

class EncoderDecoderContentSelection(tf.keras.Model):
    """ CS&P model from the thesis.

    Processes the input records, generates a content plan, which filters and organizes
    the input records and decodes output text from the contetnt planned sequence
    """
    def __init__( self
                , encoder_content_selection
                , encoder_content_planner
                , encoder_from_cp
                , text_decoder):
        """ Initialize EncoderDecoderContentSelection model

        Args:
            encoder_content_selection (model):                  one of Encoder, EncoderCS and EncoderCSBi
            encoder_content_planner   (ContentPlanDecoderCell): content_planning_decoder in the notation from thesis
            encoder_from_cp           (bidirectional lstm):     content plan encoder in the notation from thesis
            text_decoder              (model):                  one of DecoderRNNCell and DecoderRNNCellJointCopy
        """
        super().__init__()
        self._encoder_content_selection = encoder_content_selection
        self._encoder_content_planner = encoder_content_planner
        self._encoder_from_cp = encoder_from_cp
        self._text_decoder = text_decoder

    def compile( self
               , optimizer_cp
               , optimizer_txt
               , loss_fn_cp
               , loss_fn_decoder
               , scheduled_sampling_rate
               , truncation_size
               , truncation_skip_step
               , cp_training_rate = 0.1):
        """ Prepare the model for training, evaluation and prediction

        Assigns optimizers, losses, initiates training hyperparameters, sets up eager execution,
        which enables us to use different settings for training (we use graph execution during training)
        and evaluation and prediction (where we use eager execution)

            Args:
            optimizer_cp            (optimizer):    currently unused variable
            optimizer_txt           (optimizer):    optimizer used to minimize both the cp loss and the txt loss
            loss_fn_cp              (loss):         loss used on the content planning decoder outputs
            loss_fn_decoder         (loss):         loss used on the text decoder outputs
            scheduled_sampling_rate (float):        frequency at which the gold outputs from the previous time-steps are fed into the network
                                                    (number between 0 and 1, 1 means regular training)
            truncation_size         (int):          t_2 argument of TBPTT (explained in section 4.1 of the thesis)
            truncation_skip_step    (int):          t_1 argument of TBPTT (should be lower than or equal to t_2)
            cp_training_rate        (float):        number between 0 and 1, fraction of batches where we also train the content planning decoder
        """
        super().compile(run_eagerly=True)
        self._optimizer_cp = optimizer_cp
        self._optimizer_txt = optimizer_txt
        self._loss_fn_cp = loss_fn_cp
        self._loss_fn_decoder = loss_fn_decoder

        # prepare metrics reported during training
        self._train_metrics = { "accuracy_decoder" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_decoder')
                              , "accuracy_cp" :tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_cp')
                              , "loss_decoder" :tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_decoder')
                              , "loss_cp": tf.keras.metrics.SparseCategoricalCrossentropy( name='loss_cp'
                                                                                         , from_logits=True)}

        # prepare metrics reported during evaluation
        self._val_metrics = { "val_accuracy_decoder" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_decoder')
                            , "val_accuracy_cp" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_cp')
                            , "val_loss_decoder" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_decoder')
                            , "val_loss_cp" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_cp'
                                                                                            , from_logits=True)}
        self._scheduled_sampling_rate = scheduled_sampling_rate
        self._cp_training_rate = 1.0 - cp_training_rate
        self._truncation_skip_step = truncation_skip_step
        self._truncation_size = truncation_size
        self._generator = tf.random.Generator.from_non_deterministic_state()

    
    def _calc_loss( self, x, y, loss_object, selected_metrics):
        """ use only the non-pad values to calculate loss"""
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss_ = loss_object(y, x)
        mask = tf.cast(mask, loss_.dtype)
        loss_ *= mask
        for metric in self._train_metrics.values():
            if metric.name in selected_metrics:
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
            batch_data:     inputs and targets for the text decoder, content plan decoder, input tables, and
                            generated probabilities for scheduled sampling and training of content planning decoder
                            (dec_in, dec_targets, gen_or_teach, train_cp_loss, cp_in, cp_targets, tables)
            last_out:       last output of the text decoder
            initial_state:  initial state for the text decoder
        
        Returns:
            self._truncation_skip_step-th hidden state
            argmax of self._truncation_skip_step-th prediction of the network
        """
        loss_cp = 0
        loss_txt = 0
        dec_in, targets, gen_or_teach, train_cp_loss, cp_in, cp_targets, *tables = batch_data
        batch_size = cp_in.shape[0]
        final_state = None
        final_last_out = None
        cp_enc_outs = tf.TensorArray(tf.float32, size=cp_targets.shape[1])
        cp_enc_ins = tf.TensorArray(tf.int16, size=cp_targets.shape[1])
        with tf.GradientTape() as tape:
            enc_outs, *out_states = self._encoder_content_selection(tables)
            # encoder_content_selection returns 4 states, only the first two are used
            states = (out_states[0], out_states[1])
            next_input = enc_outs[:, 0, :]
            # create content plan, evaluate the loss from the 
            # gold content plan
            for t in range(cp_in.shape[1]):
                (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                      , states=states
                                                                      , training=True)
                # content_plan generation is updated only once per batch to not be affected
                # by the truncated BPTT
                if initial_state is None:
                    # the neural network is taught to predict
                    # indices shifted by 1
                    loss_cp += self._calc_loss( alignment
                                              , cp_targets[:, t]
                                              , self._loss_fn_cp
                                              , ["loss_cp", "accuracy_cp"])
                
                # prepare inputs for encoder
                # indices are shifted by 1
                # enc_outs[:, enc_outs.shape[1], :] is either
                # encoded <<EOS>> record or <<PAD>> record
                ic = tf.where(cp_targets[:, t] != 0, cp_targets[:, t] - 1, enc_outs.shape[1] - 1)
                indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)
                next_input = tf.gather_nd(enc_outs, indices)

                # the next input should be zeroed out if the indices point to the end of the table - <<EOS>> or <<PAD>> tokens
                # then the encoder_from_cp wouldn't take them into acount
                enc_outs_zeroed = tf.where(tf.expand_dims(indices[:, 1] == (enc_outs.shape[1] - 1), 1), tf.zeros(next_input.shape), next_input)
                vals = tf.gather_nd(tables[2], indices)
                cp_enc_outs = cp_enc_outs.write(t, enc_outs_zeroed)
                cp_enc_ins = cp_enc_ins.write(t, vals)

            cp_enc_outs = tf.transpose(cp_enc_outs.stack(), [1, 0, 2])
            cp_enc_ins = tf.transpose(cp_enc_ins.stack(), [1, 0])

            # encode generated content plan
            cp_enc_outs, *last_hidden_rnn = self._encoder_from_cp(cp_enc_outs, training=True)

            # prepare states and inputs for the text decoder
            if initial_state is None:
                initial_state = [ last_hidden_rnn[-1]
                                , *last_hidden_rnn ]

            if isinstance(self._text_decoder, DecoderRNNCellJointCopy):
                print("using joint copy mechanism !")
                enc_ins = tf.one_hot(tf.cast(cp_enc_ins, tf.int32), self._text_decoder._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
                aux_inputs = (cp_enc_outs, enc_ins) # value portion of the record needs to be copied
            else:
                print("using vanilla attention")
                aux_inputs = (cp_enc_outs,)
            
            # decode text from the encoded content plan
            states = initial_state
            for t in range(dec_in.shape[1]):
                if (self._truncation_skip_step is not None) and (t == self._truncation_skip_step):
                    final_state = states
                    final_last_out = last_out
                if gen_or_teach[t] > self._scheduled_sampling_rate:
                    _input = last_out
                else:
                    _input = dec_in[:, t, :]
                last_out, states = self._text_decoder( (_input, *aux_inputs)
                                                     , states=states
                                                     , training=True)
                # calculate txt loss and collect metrics
                loss_txt += self._calc_loss( last_out
                                           , targets[:, t]
                                           , self._loss_fn_decoder
                                           , ["loss_decoder", "accuracy_decoder"])
                # prepare new input for the decoder (used with (1 - scheduled_sampling_rate) probability)
                last_out = tf.expand_dims(tf.cast(tf.argmax(last_out, axis=1), tf.int16), -1)
            loss = loss_txt
            if train_cp_loss > self._cp_training_rate:
                loss += loss_cp

        variables_cp = []
        if train_cp_loss > self._cp_training_rate:
            for var in self._encoder_content_planner.trainable_variables:
                if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0'):
                    variables_cp.append(var)

        variables_txt = []
        for var in self._text_decoder.trainable_variables + \
                   self._encoder_from_cp.trainable_variables + \
                   self._encoder_content_selection.trainable_variables:
            if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0'):
                variables_txt.append(var)
        
        variables = variables_cp + variables_txt
        gradients = tape.gradient(loss, variables)
        self._optimizer_txt.apply_gradients(zip(gradients, variables))

        if (self._truncation_skip_step is None) or (self._truncation_skip_step == dec_in.shape[1]):
            final_state = states
            final_last_out = last_out

        return final_state, final_last_out


    @property
    def metrics(self):
        """returns the list of metrics that should be reset at the start and end of training epoch and evaluation"""
        return list(self._train_metrics.values()) + list(self._val_metrics.values())


    def train_step(self, batch_data):
        """ perform one train_step during model.fit

        Args:
            batch_data: data to train on in format (summaries, content_plan, *tables)
        """
        summaries, content_plan, *tables = batch_data

        # prepare inputs for the text decoder
        sums = tf.expand_dims(summaries, axis=-1)

        # scheduled sampling may force the text decoder to generate
        # from its last prediction even at the first step
        # by setting last_out = sums[:, 0] we erase differences between
        # scheduled sampling and teacher forcing at the first timestep
        last_out = sums[:, 0]
        start = 0
        length = summaries.shape[1]
        cp_length = content_plan.shape[1]
        state = None

        # train_cp_loss decides whether to train the cp decoder or not during the actual batch
        train_cp_loss = self._generator.uniform(shape=(), maxval=1.0)
        for end in range(self._truncation_size, length-1, self._truncation_skip_step):
            # gen_or_teach is the [0,1] vector which contains value for each time-step
            # if the value for the timestep is higher than self.scheduled_sampling_rate
            # the text decoder is forced to generate from its last prediction
            gen_or_teach = np.zeros(shape=(end-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)

            # prepare data for teacher forcing, scheduled sampling, cp training etc.
            truncated_data = ( sums[:, start:end, :]
                             , summaries[:, start+1:end+1]
                             , tf.convert_to_tensor(gen_or_teach)
                             , tf.convert_to_tensor(train_cp_loss)
                             , content_plan[:, :cp_length - 1]
                             , content_plan[:, 1:cp_length]
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
            
            # prepare data for teacher forcing, scheduled sampling, cp training etc.
            truncated_data = ( sums[:, start:length-1, :]
                             , summaries[:, start+1:length]
                             , tf.convert_to_tensor(gen_or_teach)
                             , train_cp_loss
                             , content_plan[:, :cp_length - 1]
                             , content_plan[:, 1:cp_length]
                             , *tables)
            # run the backpropagation on truncated sequence
            state, last_out = self.bppt_step( truncated_data
                                            , last_out
                                            , initial_state=state)
        return dict([(metric.name, metric.result()) for metric in self._train_metrics.values()])

    def test_step(self, batch_data):
        """ perform one test_step during model.evaluate

        Args:
            batch_data: data to train on in format (summaries, content_plan, *tables)
        """
        summaries, content_plan, *tables = batch_data
        # prepare summaries
        max_sum_size = summaries.shape[1] - 1
        dec_in = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]
        targets = summaries[:, 1:max_sum_size+1]

        # prepare content plans
        cp_length = content_plan.shape[1]
        cp_in = content_plan[:, :cp_length-1]
        cp_targets = content_plan[:, 1:cp_length]

        batch_size = cp_in.shape[0]
        cp_enc_outs = tf.TensorArray(tf.float32, size=cp_targets.shape[1])
        cp_enc_ins = tf.TensorArray(tf.int16, size=cp_targets.shape[1])
        enc_outs, *out_states = self._encoder_content_selection(tables)
        states = (out_states[0], out_states[1])

        next_input = enc_outs[:, 0, :]
        # create content plan, evaluate the loss from the 
        # gold content plan
        for t in range(cp_in.shape[1]):
            (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                  , states=states
                                                                  , training=False)
            
            # calculate the metrics on the generated content plans
            mask = tf.math.logical_not(tf.math.equal(cp_targets[:, t], 0))
            for metric in self._val_metrics.values():
                if metric.name in ["accuracy_cp", "loss_cp"]:
                    metric.update_state( cp_targets[:, t]
                                       , alignment
                                       , sample_weight=mask )
            
            # prepare inputs for encoder
            # indices are shifted by 1
            # enc_outs[:, enc_outs.shape[1], :] is either
            # encoded <<EOS>> record or <<PAD>> record
            ic = tf.where(cp_targets[:, t] != 0, cp_targets[:, t] - 1, enc_outs.shape[1] - 1)
            indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)
            next_input = tf.gather_nd(enc_outs, indices)

            # the next input should be zeroed out if the indices point to the end of the table - <<EOS>> or <<PAD>> tokens
            # then the encoder_from_cp wouldn't take them into acount
            enc_outs_zeroed = tf.where(tf.expand_dims(indices[:, 1] == (enc_outs.shape[1] - 1), 1), tf.zeros(next_input.shape), next_input)
            vals = tf.gather_nd(tables[2], indices)
            cp_enc_outs = cp_enc_outs.write(t, enc_outs_zeroed)
            cp_enc_ins = cp_enc_ins.write(t, vals)

        cp_enc_outs = tf.transpose(cp_enc_outs.stack(), [1, 0, 2])
        cp_enc_ins = tf.transpose(cp_enc_ins.stack(), [1, 0])

        # encode generated content plan
        cp_enc_outs, *last_hidden_rnn = self._encoder_from_cp(cp_enc_outs, training=False)

        # prepare states and inputs for the text decoder
        if isinstance(self._text_decoder, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(cp_enc_ins, tf.int32), self._text_decoder._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (cp_enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (cp_enc_outs,)
        
        # decode text from the encoded content plan
        states = [ last_hidden_rnn[-1], *last_hidden_rnn ]
        for t in range(dec_in.shape[1]):
            _input = dec_in[:, t, :]
            last_out, states = self._text_decoder( (_input, *aux_inputs)
                                                 , states=states
                                                 , training=False)

            mask = tf.math.logical_not(tf.math.equal(targets[:, t], 0))
            for metric in self._val_metrics.values():
                if metric.name in ["accuracy_decoder", "loss_decoder"]:
                    metric.update_state( targets[:, t]
                                       , last_out
                                       , sample_weight=mask )
        return dict([(metric.name, metric.result()) for metric in self._val_metrics.values()])

    def predict_step(self, batch_data):
        """ perform one predict_step during model.predict

        Args:
            batch_data: data to train on in format (summaries, content_plan, *tables)
        """
        summaries, content_plan, *tables = batch_data
        # prepare summaries
        max_sum_size = summaries.shape[1] - 1
        dec_in = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]

        batch_size = content_plan.shape[0]
        cp_enc_outs = tf.TensorArray(tf.float32, size=content_plan.shape[1])
        cp_enc_ins = tf.TensorArray(tf.int16, size=content_plan.shape[1])
        cp_cp_ix = tf.TensorArray(tf.int32, size=content_plan.shape[1])
        enc_outs, *out_states = self._encoder_content_selection(tables)
        states = (out_states[0], out_states[1])

        # the first input to the encoder_content_planner is 0th record
        # zeroth record is the <<BOS>> record
        next_input = enc_outs[:, 0, :]

        # create content plan
        # next input of the encoder_content_planner is its last output
        for t in range(content_plan.shape[1] - 1):
            # indices are shifted by 1
            # enc_outs[:, enc_outs.shape[1], :] is either
            # encoded <<EOS>> record or <<PAD>> record
            (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                  , states=states
                                                                  , training=False)
            
            # prepare next_input and gather inputs for the encoder

            # get max indices
            max_alignment = tf.argmax(alignment, axis=-1, output_type=tf.dtypes.int32)
            ic = tf.where(max_alignment != 0, max_alignment - 1, enc_outs.shape[1] - 1)
            indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)

            # get correct values from tables
            vals = tf.gather_nd(tables[2], indices)
            next_input = tf.gather_nd(enc_outs, indices)

            # save for decoder
            cp_cp_ix = cp_cp_ix.write(t, ic)
            enc_outs_zeroed = tf.where(tf.expand_dims(indices[:, 1] == (enc_outs.shape[1] - 1), 1), tf.zeros(next_input.shape), next_input)
            cp_enc_outs = cp_enc_outs.write(t, enc_outs_zeroed)
            cp_enc_ins = cp_enc_ins.write(t, vals)

        cp_enc_outs = tf.transpose(cp_enc_outs.stack(), [1, 0, 2])
        cp_enc_ins = tf.transpose(cp_enc_ins.stack(), [1, 0])
        cp_cp_ix = tf.transpose(cp_cp_ix.stack(), [1, 0])

        # encode generated content plan
        cp_enc_outs, *last_hidden_rnn = self._encoder_from_cp(cp_enc_outs, training=False)

        # prepare states and inputs for the text decoder
        if isinstance(self._text_decoder, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(cp_enc_ins, tf.int32), self._text_decoder._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (cp_enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (cp_enc_outs,)

        # decode text from the encoded content plan
        states = [ last_hidden_rnn[-1], *last_hidden_rnn ]
        # zeroth token is the <<BOS>> token
        _input = dec_in[:, 0, :]
        result_preds = np.zeros(summaries.shape, dtype=np.int)
        for t in range(dec_in.shape[1]):
            last_out, states = self._text_decoder( (_input, *aux_inputs)
                                                 , states=states
                                                 , training=False)

            predicted = tf.argmax(last_out, axis=1).numpy()
            result_preds[:, t] = predicted
            _input = tf.expand_dims(predicted, axis=1)
        self.last_content_plan = cp_cp_ix
        return result_preds

from neural_nets.cp_model import EncoderDecoderContentSelection
from neural_nets.layers import DecoderRNNCellJointCopy
import numpy as np

class GreedyAdapter(tf.keras.Model):
    """ GreedyAdapter for the EncoderDecoderContentSelection

    since we save models to tf.train.checkpoints, we do not want to experiment
    whether the checkpoint would load the saved model into its changed counterpart or not

    Therefore for experiments with masking the content plans (described in section 5.5.1) we
    create new adapter which steals the internals of the EncoderDecoderContentSelection
    and greedily generates the outputs
    """
    def __init__( self
                , encoder_decoder : EncoderDecoderContentSelection
                , max_cp_size
                , from_gold_cp):
        """ Initialize GreedyAdapter on the EncoderDecoderContentSelection

        Args:
            encoder_decoder:    trained model, from which we steal its internals
            max_cp_size:        maximal size of the generated content_plan
            from_gold_cp:       whether to generate content plans and decode text from the generated
                                content plans or decode text from the gold content plans
        """
        super().__init__()
        self._encoder = ContentPlanDecoder(encoder_decoder, max_cp_size)
        self._text_decoder = encoder_decoder._text_decoder
        self._from_gold_cp = from_gold_cp

    def compile(self):
        super().compile(run_eagerly=True)
        self._encoder.compile()
    
    def predict_step(self, batch_data):
        summaries, content_plan, *tables = batch_data
        # prepare summaries
        max_sum_size = summaries.shape[1] - 1
        dec_in = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]

        if not self._from_gold_cp:
            cp_enc_outs, cp_enc_ins, last_hidden_rnn = self._encoder(tables)        
        else:
            # prepare content plans
            cp_length = content_plan.shape[1]
            cp_in = content_plan[:, :cp_length-1]
            cp_targets = content_plan[:, 1:cp_length]
            cp_enc_outs = tf.TensorArray(tf.float32, size=cp_targets.shape[1])
            cp_enc_ins = tf.TensorArray(tf.int16, size=cp_targets.shape[1])
            enc_outs, *out_states = self._encoder._encoder_content_selection(tables)
            # create content plan, evaluate the loss from the 
            # gold content plan
            for t in range(cp_in.shape[1]):                
                # prepare inputs for encoder
                # indices are shifted by 1
                # enc_outs[:, enc_outs.shape[1], :] is either
                # encoded <<EOS>> record or <<PAD>> record
                ic = tf.where(cp_targets[:, t] != 0, cp_targets[:, t] - 1, enc_outs.shape[1] - 1)
                indices = tf.stack([tf.range(enc_outs.shape[0]), tf.cast(ic, tf.int32)], axis=1)
                next_input = tf.gather_nd(enc_outs, indices)

                # the next input should be zeroed out if the indices point to the end of the table - <<EOS>> or <<PAD>> tokens
                # then the encoder_from_cp wouldn't take them into acount
                enc_outs_zeroed = tf.where(tf.expand_dims(indices[:, 1] == (enc_outs.shape[1] - 1), 1), tf.zeros(next_input.shape), next_input)
                vals = tf.gather_nd(tables[2], indices)
                cp_enc_outs = cp_enc_outs.write(t, enc_outs_zeroed)
                cp_enc_ins = cp_enc_ins.write(t, vals)

            cp_enc_outs = tf.transpose(cp_enc_outs.stack(), [1, 0, 2])
            cp_enc_ins = tf.transpose(cp_enc_ins.stack(), [1, 0])

            # encode generated content plan
            cp_enc_outs, *last_hidden_rnn = self._encoder._encoder_from_cp(cp_enc_outs, training=False)

        # prepare states and inputs for the text decoder
        if isinstance(self._text_decoder, DecoderRNNCellJointCopy):
            enc_ins = tf.one_hot(tf.cast(cp_enc_ins, tf.int32), self._text_decoder._word_vocab_size) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            aux_inputs = (cp_enc_outs, enc_ins) # value portion of the record needs to be copied
        else:
            aux_inputs = (cp_enc_outs,)

        # decode text from the encoded content plan
        states = [ last_hidden_rnn[-1], *last_hidden_rnn ]
        # zeroth token is the <<BOS>> token
        _input = dec_in[:, 0, :]
        result_preds = np.zeros(summaries.shape, dtype=np.int)
        for t in range(dec_in.shape[1]):
            last_out, states = self._text_decoder( (_input, *aux_inputs)
                                                 , states=states
                                                 , training=False)

            predicted = tf.argmax(last_out, axis=1).numpy()
            result_preds[:, t] = predicted
            _input = tf.expand_dims(predicted, axis=1)
        return result_preds

class ContentPlanDecoder(tf.keras.Model):
    """ Wrapper around Content Planning part of the EncoderDecoderContentSelection
    
    Greedily generates content plan.
    """
    def __init__( self
                , encoder_decoder : EncoderDecoderContentSelection
                , max_cp_size):
        """ Initialize GreedyAdapter on the EncoderDecoderContentSelection

        Args:
            encoder_decoder:    trained model, from which we steal its internals
            max_cp_size:        maximal size of the generated content_plan
        """
        super(ContentPlanDecoder, self).__init__()
        self._encoder_content_selection = encoder_decoder._encoder_content_selection
        self._encoder_content_planner = encoder_decoder._encoder_content_planner
        self._encoder_from_cp = encoder_decoder._encoder_from_cp
        self._max_cp_size = max_cp_size

    def compile(self):
        """ we compile the model to enable eager execution """
        super(ContentPlanDecoder, self).compile(run_eagerly=True)
    
    def call(self, tables):
        # prepare summaries
        PRUN = 60
        batch_size = tables[0].shape[0]
        cp_enc_outs = tf.TensorArray(tf.float32, size=self._max_cp_size)
        cp_enc_ins = tf.TensorArray(tf.int16, size=self._max_cp_size)
        cp_cp_ix = tf.TensorArray(tf.int32, size=self._max_cp_size)
        enc_outs, *out_states = self._encoder_content_selection(tables)
        states = (out_states[0], out_states[1])

        # the first input to the encoder_content_planner is 0th record
        # zeroth record is the <<BOS>> record
        next_input = enc_outs[:, 0, :]
        mask = tf.ones(shape=next_input.shape[0], dtype=tf.bool)

        # create content plan
        # next input of the encoder_content_planner is its last output
        for t in range(self._max_cp_size - PRUN):
            # we want to mask all the outputs after generation of the EOS token, which 
            # happens to have value of 2
            (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                  , states=states
                                                                  , training=False)
            
            # prepare next_input and gather inputs for the encoder

            # get max indices
            max_alignment = tf.argmax(alignment, axis=-1, output_type=tf.dtypes.int32)

            max_alignment = max_alignment * tf.cast(mask, dtype=tf.int32) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            mask = tf.math.logical_not(tf.math.logical_or(tf.math.logical_not(mask), max_alignment == 2))

            # change the value of max alignment so that it points to one record lower
            # if <<EOS>> was already generated point to the last record of the table, the <<PAD>>
            # record
            ic = tf.where(max_alignment != 0, max_alignment - 1, enc_outs.shape[1] - 1)
            ic = tf.where(ic != 1, ic, ic + 1)
            if t == self._max_cp_size - PRUN - 1:
                ic = tf.where( mask != False
                             , tf.ones(mask.shape, dtype=tf.dtypes.int32) * 2
                             , tf.cast(mask, dtype=tf.dtypes.int32)) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)

            # get correct values from tables
            vals = tf.gather_nd(tables[2], indices)
            next_input = tf.gather_nd(enc_outs, indices)

            # save for decoder
            cp_cp_ix = cp_cp_ix.write(t, ic)
            enc_outs_zeroed = tf.where(tf.expand_dims(indices[:, 1] == (enc_outs.shape[1] - 1), 1), tf.zeros(next_input.shape), next_input)
            cp_enc_outs = cp_enc_outs.write(t, enc_outs_zeroed)
            cp_enc_ins = cp_enc_ins.write(t, vals)

        cp_enc_outs = tf.transpose(cp_enc_outs.stack(), [1, 0, 2])
        cp_enc_ins = tf.transpose(cp_enc_ins.stack(), [1, 0])
        cp_cp_ix = tf.transpose(cp_cp_ix.stack(), [1, 0])

        # encode generated content plan
        cp_enc_outs, *last_hidden_rnn = self._encoder_from_cp(cp_enc_outs, training=False)
        return cp_enc_outs, cp_enc_ins, last_hidden_rnn
