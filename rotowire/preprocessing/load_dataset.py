import tensorflow as tf

def load_tf_record_dataset( path
                          , batch_size
                          , shuffle
                          , preprocess_table_size
                          , preprocess_summary_size):
    bound_1 = preprocess_summary_size
    bound_2 = bound_1 + preprocess_table_size
    bound_3 = bound_2 + preprocess_table_size
    bound_4 = bound_3 + preprocess_table_size

    # code from https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
    def read_map_fn(x):
        xp = tf.io.parse_tensor(x, tf.int16)
        xp.set_shape([bound_4 + preprocess_table_size])
        #      summary       types                  entities               values                 home/away
        return xp[:bound_1], xp[bound_1 : bound_2], xp[bound_2 : bound_3], xp[bound_3 : bound_4], xp[bound_4:]

    data = tf.data.TFRecordDataset(path).map(read_map_fn)
    print("data loading : created dataset!", flush=True)
    BUFFER_SIZE = 1000
    print("data loading : created dataset!", flush=True)
    if shuffle:
        data = data.shuffle(BUFFER_SIZE)
    print("data loading : shuffled dataset!", flush=True)
    data = data.batch(batch_size, drop_remainder=True)
    data.cache()
    print("data loading : batched dataset!", flush=True)
    steps = len(list(data.as_numpy_iterator()))
    print(f"data loading : {steps} steps per epoch!")
    return data, steps