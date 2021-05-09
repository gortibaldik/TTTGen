import tensorflow as tf

from preprocessing.utils import OccurrenceDict, create_ha_vocab, create_tp_vocab

def load_values_from_config( config_path
                           , load_cp : bool = False):
    with open(config_path, 'r') as f:
        contents = f.read().strip().split('\n')
    if (load_cp and len(contents) != 3) or (not load_cp and len(contents) != 2):
        raise RuntimeError(f"Invalid config dir {config_path}, len(contents) : {len(contents)}")
    max_table_size = int(contents[0])
    max_summary_size = int(contents[1]) + 2 # eos and bos tokens
    if load_cp:
        max_cp_size = int(contents[2])
    else:
        max_cp_size = 0
    return max_table_size, max_summary_size, max_cp_size

def load_tf_record_dataset( path
                          , vocab_path
                          , batch_size : int
                          , shuffle : bool
                          , preprocess_table_size : int
                          , preprocess_summary_size : int
                          , preprocess_cp_size : int = None
                          , with_content_plans : bool = False):
    bound_1 = preprocess_summary_size
    bound_2 = bound_1 + preprocess_table_size
    bound_3 = bound_2 + preprocess_table_size
    bound_4 = bound_3 + preprocess_table_size

    if with_content_plans:
        cpbound_1 = preprocess_summary_size
        cpbound_2 = cpbound_1 + preprocess_cp_size
        cpbound_3 = cpbound_2 + preprocess_table_size
        cpbound_4 = cpbound_3 + preprocess_table_size
        cpbound_5 = cpbound_4 + preprocess_table_size

    # code from https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
    def read_map_fn(x):
        xp = tf.io.parse_tensor(x, tf.int16)
        xp.set_shape([bound_4 + preprocess_table_size])
        #      summary       types                  entities               values                 home/away
        return xp[:bound_1], xp[bound_1 : bound_2], xp[bound_2 : bound_3], xp[bound_3 : bound_4], xp[bound_4:]

    def read_map_fn_cp(x):
        xp = tf.io.parse_tensor(x, tf.int16)
        xp.set_shape([cpbound_5 + preprocess_table_size])
        # summary content_plan types entities values home/away
        return xp[:cpbound_1] \
             , xp[cpbound_1: cpbound_2] \
             , xp[cpbound_2:cpbound_3] \
             , xp[cpbound_3:cpbound_4] \
             , xp[cpbound_4:cpbound_5] \
             , xp[cpbound_5:] 

    # prepare dataset
    if not with_content_plans:
        data = tf.data.TFRecordDataset(path).map(read_map_fn)
    else:
        data = tf.data.TFRecordDataset(path).map(read_map_fn_cp)
        # filtering out batch, where there is too big table
        # TODO: make preprocessing better !
        def filter_fn(summaries, cp, *tables):
            return not tf.reduce_any(cp == tf.cast(tf.ones(cp.shape) * 692, tf.int16)) # pylint: disable=no-value-for-parameter
        data = data.filter(filter_fn)

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

    # prepare token vocab
    vocab = OccurrenceDict.load(vocab_path)
    tk_to_ix = vocab.to_dict()

    # prepare type vocab
    tp_vocab = create_tp_vocab()
    tp_to_ix = tp_vocab.to_dict()

    # prepare home/away vocab
    ha_vocab = create_ha_vocab()
    ha_to_ix = ha_vocab.to_dict()

    pad_token = tk_to_ix[vocab.get_pad()]
    if pad_token != tp_to_ix[tp_vocab.get_pad()] or pad_token != ha_to_ix[ha_vocab.get_pad()]:
        raise RuntimeError("Different pad values in the vocab!")
    bos_token = tk_to_ix[vocab.get_bos()]
    eos_token = tk_to_ix[vocab.get_eos()]
    return data, steps, tk_to_ix, tp_to_ix, ha_to_ix, pad_token, bos_token, eos_token