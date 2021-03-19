from enum import Enum
import numpy as np
import tensorflow as tf
import os
import argparse

'''
MODEL_SET_DATA_DIR needs to be defined before first use of Enum ModelSet,
then the valid data could be found, otherwise the behavior is undefined
'''
MODEL_SET_DATA_DIR = "/content/drive/My Drive/sequence_aware"


class ModelSet(Enum):
    train = ['train/train.summary.id',
             'train/train.box.val.id',
             'train/train.box.lab.id',
             'train/train.box.pos',
             'train/train.box.rpos',
             'train/train.box.val',
             'train/train.box.lab']
    test =  ['test/test.summary.id',
             'test/test.box.val.id',
             'test/test.box.lab.id',
             'test/test.box.pos',
             'test/test.box.rpos',
             'test/test.box.val',
             'test/test.box.lab' ]
    val =   ['valid/valid.summary.id',
             'valid/valid.box.val.id',
             'valid/valid.box.lab.id',
             'valid/valid.box.pos',
             'valid/valid.box.rpos',
             'valid/valid.box.val',
             'valid/valid.box.lab']


def load_npy(path : str):
    print(f"data_loading : Loading npy archive from {path}", flush=True)
    return np.load(path)


def load_dataset(model_set, batch_size, shuffle : bool = True, num_examples = None):
    summary_path, text_path, field_path, pos_path, rpos_path, vals_path, labs_path = model_set.value
    summaries = load_npy(os.path.join(MODEL_SET_DATA_DIR, summary_path))
    texts = load_npy(os.path.join(MODEL_SET_DATA_DIR, text_path))
    fields = load_npy(os.path.join(MODEL_SET_DATA_DIR, field_path))
    poses = load_npy(os.path.join(MODEL_SET_DATA_DIR, pos_path))
    rposes = load_npy(os.path.join(MODEL_SET_DATA_DIR, rpos_path))

    if num_examples is not None:
        summaries = summaries[:num_examples]
        texts = texts[:num_examples]
        fields = fields[:num_examples]
        poses = poses[:num_examples]
        rposes = rposes[:num_examples]


    summaries = tf.convert_to_tensor(summaries)
    texts = tf.convert_to_tensor(texts)
    fields = tf.convert_to_tensor(fields)
    poses = tf.convert_to_tensor(poses)
    rposes = tf.convert_to_tensor(rposes)

    steps_per_epoch = summaries.shape[0] // batch_size
    for d in [summaries, texts, fields, poses, rposes]:
        print(f"data loading : loaded data with shape : {d.shape}", flush=True)
    BUFFER_SIZE = len(summaries)

    data = tf.data.Dataset.from_tensor_slices(
        (summaries,
         texts,
         fields,
         poses,
         rposes)
    )
    print("data loading : created dataset!", flush=True)
    if shuffle:
        data = data.shuffle(BUFFER_SIZE)
    print("data loading : shuffled dataset!", flush=True)
    data = data.batch(batch_size, drop_remainder=True)
    print("data loading : batched dataset!", flush=True)
    return data, steps_per_epoch


def convert_to_path(set_type : str):
    return os.path.join(MODEL_SET_DATA_DIR, "data_" + set_type + ".tfrecord")


def convert_to_tfrecord( model_set
                       , set_name : str):
    summary_path, text_path, field_path, pos_path, rpos_path, vals_path, labs_path = model_set.value
    summaries = load_npy(os.path.join(MODEL_SET_DATA_DIR, summary_path))
    texts = load_npy(os.path.join(MODEL_SET_DATA_DIR, text_path))
    fields = load_npy(os.path.join(MODEL_SET_DATA_DIR, field_path))
    poses = load_npy(os.path.join(MODEL_SET_DATA_DIR, pos_path))
    rposes = load_npy(os.path.join(MODEL_SET_DATA_DIR, rpos_path))

    summaries = tf.convert_to_tensor(summaries, dtype=tf.int16)
    texts = tf.convert_to_tensor(texts, dtype=tf.int16)
    fields = tf.convert_to_tensor(fields, dtype=tf.int16)
    poses = tf.convert_to_tensor(poses, dtype=tf.int16)
    rposes = tf.convert_to_tensor(rposes, dtype=tf.int16)

    for d in [summaries, texts, fields, poses, rposes]:
        print(f"data loading : loaded data with shape : {d.shape}", flush=True)

    data = tf.data.Dataset.from_tensor_slices(
        (summaries,
         texts,
         fields,
         poses,
         rposes)
    )

    # code from https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
    def write_map_fn(summary, text, field, pos, rpos):
        return tf.io.serialize_tensor(tf.concat([summary, text, field, pos, rpos], axis=-1))
    data = data.map(write_map_fn)
    data_path = convert_to_path(set_name)
    writer = tf.data.experimental.TFRecordWriter(data_path)
    writer.write(data)


def load_tf_record_dataset( set_name
                          , batch_size
                          , shuffle : bool = True
                          , preprocess_table_size : int = 100
                          , preprocess_summary_size : int = 76):
    # code from https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
    bound_1 = preprocess_summary_size
    bound_2 = preprocess_summary_size + preprocess_table_size
    bound_3 = preprocess_summary_size + preprocess_table_size * 2
    bound_4 = preprocess_summary_size + preprocess_table_size * 3

    def read_map_fn(x):
        xp = tf.io.parse_tensor(x, tf.int16)
        xp.set_shape([preprocess_summary_size + 4 * preprocess_table_size])
        #      summary       text                   field                  pos                    rpos
        return xp[:bound_1], xp[bound_1 : bound_2], xp[bound_2 : bound_3], xp[bound_3 : bound_4], xp[bound_4:]

    data_path = convert_to_path(set_name)
    data = tf.data.TFRecordDataset(data_path).map(read_map_fn)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("type_of_op", type=str, default="create"
            , help="<create|check>")
    args = parser.parse_args()
    MODEL_SET_DATA_DIR = "processed_data/"
    if args.type_of_op == "check":
        load_tf_record_dataset("train"
                              , batch_size=32
                              , shuffle=True)
        load_tf_record_dataset("valid"
                              , batch_size=32
                              , shuffle=False)
    elif args.type_of_op == "create":
        convert_to_tfrecord(ModelSet.train, "train")
        convert_to_tfrecord(ModelSet.val, "valid")
    else :
        print("UNKNOWN OP", flush=True)
