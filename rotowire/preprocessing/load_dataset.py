import tensorflow as tf

from preprocessing.utils import OccurrenceDict
from preprocessing.preprocessing import create_tp_vocab, create_ha_vocab

def load_values_from_config(config_path):
    with open(config_path, 'r') as f:
        contents = f.read().strip().split('\n')
    if len(contents) != 2:
        raise RuntimeError("Invalid config dir")
    max_table_size = int(contents[0])
    max_summary_size = int(contents[1]) + 2 # eos and bos tokens
    return max_table_size, max_summary_size

def load_tf_record_dataset( path
                          , vocab_path
                          , batch_size : int
                          , shuffle : bool
                          , preprocess_table_size : int
                          , preprocess_summary_size : int):
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

    # prepare dataset
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