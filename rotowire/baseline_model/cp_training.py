""" just a dummy copied from my colab notebook, unresolved imports """

def loss_function( x
                 , y
                 , loss_object):
    """ use only the non-pad values """
    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss_object(y, x)
    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def pass_through( tables
                , content_plan
                , dec_input
                , dec_target
                , encoder
                , decoder_cp
                , decoder
                , loss_object
                , optimizer):
    batch_size = content_plan.shape[0]
    loss = 0
    contexts = tf.TensorArray(tf.float32, size=content_plan.shape[1])
    with tf.GradientTape() as tape:
        enc_outs, avg = encoder(tables)
        states = (avg, avg)
        for t in range(content_plan.shape[1]):
            indices = tf.stack([tf.range(batch_size), tf.cast(content_plan[:, t], tf.int32)], axis=1)
            next_input = tf.gather_nd(enc_outs, indices)
            (context, alignment), states = decoder_cp((next_input, enc_outs), states=states, training=True)
            loss += loss_function( alignment
                                , content_plan[:, t]
                                , loss_object)
            contexts.write(t, context)
        
        contexts = tf.transpose(contexts.stack(), [1,0,2])
        states = (avg, avg, avg, avg, avg)
        for t in range(dec_input.shape[1]):
            input = dec_input[:, t, :]
            last_out, states = decoder((input, contexts), states=states, training=True)
            loss += loss_function( last_out, dec_target[:, t], loss_object)
        
        print(contexts.shape)
        print(content_plan.shape)
    
    variables = encoder.trainable_variables + decoder_cp.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))