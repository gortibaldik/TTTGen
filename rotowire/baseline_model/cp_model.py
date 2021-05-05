import tensorflow as tf
import numpy as np
from .layers import DecoderRNNCellJointCopy

class EncoderDecoderContentSelection(tf.keras.Model):
    def __init__( self
                , encoder_content_selection
                , encoder_content_planner
                , encoder_from_cp
                , text_decoder):
        super(EncoderDecoderContentSelection, self).__init__()
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
               , truncation_skip_step):
        super(EncoderDecoderContentSelection, self).compile(run_eagerly=True)
        self._optimizer_cp = optimizer_cp
        self._optimizer_txt = optimizer_txt
        self._loss_fn_cp = loss_fn_cp
        self._loss_fn_decoder = loss_fn_decoder
        self._train_metrics = { "accuracy_decoder" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_decoder')
                              , "accuracy_cp" :tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_cp')
                              , "loss_decoder" :tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_decoder')
                              , "loss_cp": tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_cp')}
        self._val_metrics = { "val_accuracy_decoder" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_decoder')
                            , "val_accuracy_cp" : tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_cp')
                            , "val_loss_decoder" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_decoder')
                            , "val_loss_cp" : tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_cp')}
        self._scheduled_sampling_rate = scheduled_sampling_rate
        self._truncation_skip_step = truncation_skip_step
        self._truncation_size = truncation_size
        self._generator = tf.random.Generator.from_non_deterministic_state()
    
    def _calc_loss( self, x, y, loss_object, selected_metrics):
        """ use only the non-pad values """
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
        loss_cp = 0
        loss_txt = 0
        dec_in, targets, gen_or_teach, cp_in, cp_targets, *tables = batch_data
        batch_size = cp_in.shape[0]
        final_state = None
        final_last_out = None
        cp_enc_outs = tf.TensorArray(tf.float32, size=cp_targets.shape[1])
        cp_enc_ins = tf.TensorArray(tf.int16, size=cp_targets.shape[1])
        with tf.GradientTape() as tape:
            enc_outs, avg = self._encoder_content_selection(tables)
            states = (avg, avg)
            next_input = enc_outs[:, 0, :]
            # create content plan, evaluate the loss from the 
            # gold content plan
            for t in range(cp_in.shape[1]):
                (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                      , states=states
                                                                      , training=True)
                
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
                potential_next_input = tf.gather_nd(enc_outs, indices)

                # the next input should be zeroed out if the indices point to the end of the table - <<EOS>> or <<PAD>> tokens
                # then the encoder_from_cp wouldn't take them into acount
                next_input = tf.where(tf.expand_dims(indices[:, 1] == enc_outs.shape[1], 1), tf.zeros(potential_next_input.shape), potential_next_input)
                vals = tf.gather_nd(tables[2], indices)
                cp_enc_outs = cp_enc_outs.write(t, next_input)
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
                loss_txt += self._calc_loss( last_out
                                           , targets[:, t]
                                           , self._loss_fn_decoder
                                           , ["loss_decoder", "accuracy_decoder"])
                last_out = tf.expand_dims(tf.cast(tf.argmax(last_out, axis=1), tf.int16), -1)
            loss = loss_cp + loss_txt

        variables_cp = []
        for var in self._encoder_content_planner.trainable_variables + \
                   self._encoder_content_selection.trainable_variables:
            if (initial_state is None) or (var.name != 'encoder/linear_transform/kernel:0'):
                variables_cp.append(var)

        variables_txt = []
        for var in self._text_decoder.trainable_variables + \
                   self._encoder_from_cp.trainable_variables:
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
        return list(self._train_metrics.values()) + list(self._val_metrics.values())


    def train_step(self, batch_data):
        summaries, content_plan, *tables = batch_data
        sums = tf.expand_dims(summaries, axis=-1)
        last_out = sums[:, 0]
        start = 0
        length = summaries.shape[1]
        cp_length = content_plan.shape[1]
        state = None
        for end in range(self._truncation_size, length-1, self._truncation_skip_step):
            gen_or_teach = np.zeros(shape=(end-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)
            # create data for teacher forcing
            truncated_data = ( sums[:, start:end, :]
                             , summaries[:, start+1:end+1]
                             , tf.convert_to_tensor(gen_or_teach)
                             , content_plan[:, :cp_length - 1]
                             , content_plan[:, 1:cp_length]
                             , *tables)
            state, last_out =  self.bppt_step( truncated_data
                                             , last_out
                                             , initial_state=state)
            start += self._truncation_skip_step
        # finish the truncated bppt if the truncation_size cannot divide properly
        # the length of sequence
        if (length - self._truncation_size) % self._truncation_skip_step != 0:
            gen_or_teach = np.zeros(shape=(length-1-start))
            for i in range(len(gen_or_teach)):
                gen_or_teach[i] = self._generator.uniform(shape=(), maxval=1.0)
            truncated_data = ( sums[:, start:length-1, :]
                             , summaries[:, start+1:length]
                             , tf.convert_to_tensor(gen_or_teach)
                             , content_plan[:, :cp_length - 1]
                             , content_plan[:, 1:cp_length]
                             , *tables)
            state, last_out = self.bppt_step( truncated_data
                                            , last_out
                                            , initial_state=state)
        return dict([(metric.name, metric.result()) for metric in self._train_metrics.values()])

    def test_step(self, batch_data):
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
        enc_outs, avg = self._encoder_content_selection(tables)
        states = (avg, avg)

        # create content plan, evaluate the loss from the 
        # gold content plan
        for t in range(cp_in.shape[1]):
            # indices are shifted by 1
            # enc_outs[:, enc_outs.shape[1], :] is either
            # encoded <<EOS>> record or <<PAD>> record
            ic = tf.where(cp_in[:, t] != 0, cp_in[:, t] - 1, enc_outs.shape[1] - 1)
            indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)
            next_input = tf.gather_nd(enc_outs, indices)
            (_, alignment), states = self._encoder_content_planner( (next_input, enc_outs)
                                                                  , states=states
                                                                  , training=False)
            
            mask = tf.math.logical_not(tf.math.equal(cp_targets[:, t], 0))
            for metric in self._val_metrics.values():
                if metric.name in ["accuracy_cp", "loss_cp"]:
                    metric.update_state( cp_targets[:, t]
                                       , alignment
                                       , sample_weight=mask )
            
            # prepare inputs for encoder
            ic = tf.where(cp_targets[:, t] != 0, cp_targets[:, t] - 1, enc_outs.shape[1] - 1)
            indices = tf.stack([tf.range(batch_size), tf.cast(ic, tf.int32)], axis=1)
            vals = tf.gather_nd(tables[2], indices)
            encs = tf.gather_nd(enc_outs, indices)
            cp_enc_outs = cp_enc_outs.write(t, encs)
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
        summaries, content_plan, *tables = batch_data
        # prepare summaries
        max_sum_size = summaries.shape[1] - 1
        dec_in = tf.expand_dims(summaries, axis=-1)[:, :max_sum_size, :]

        batch_size = content_plan.shape[0]
        cp_enc_outs = tf.TensorArray(tf.float32, size=content_plan.shape[1])
        cp_enc_ins = tf.TensorArray(tf.int16, size=content_plan.shape[1])
        cp_cp_ix = tf.TensorArray(tf.int32, size=content_plan.shape[1])
        enc_outs, avg = self._encoder_content_selection(tables)
        states = (avg, avg)

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
            cp_enc_outs = cp_enc_outs.write(t, next_input)
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
            dec_in = tf.expand_dims(predicted, axis=1)
        self.last_content_plan = cp_cp_ix
        return result_preds