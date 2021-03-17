from enum import Enum
import numpy as np
import tensorflow as tf
import os

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


def load_dataset(model_set, batch_size, shuffle : bool = True, num_examples = None):
    summary_path, text_path, field_path, pos_path, rpos_path, vals_path, labs_path = model_set.value
    summaries = np.load(os.path.join(MODEL_SET_DATA_DIR, summary_path))
    texts = np.load(os.path.join(MODEL_SET_DATA_DIR, text_path))
    fields = np.load(os.path.join(MODEL_SET_DATA_DIR, field_path))
    poses = np.load(os.path.join(MODEL_SET_DATA_DIR, pos_path))
    rposes = np.load(os.path.join(MODEL_SET_DATA_DIR, rpos_path))

    if num_examples is not None:
        summaries = summaries[:num_examples]
        texts = texts[:num_examples]
        fields = fields[:num_examples]
        poses = poses[:num_examples]
        rposes = rposes[:num_examples]

    steps_per_epoch = summaries.shape[0] // batch_size
    for d in [summaries, texts, fields, poses, rposes]:
        print(d.shape)
    BUFFER_SIZE = len(summaries)

    data = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(summaries),
         tf.convert_to_tensor(texts),
         tf.convert_to_tensor(fields),
         tf.convert_to_tensor(poses),
         tf.convert_to_tensor(rposes))
    )
    if shuffle:
        data = data.shuffle(BUFFER_SIZE)
    data = data.batch(batch_size, drop_remainder=True)
    return data, steps_per_epoch


if __name__ == "__main__":
    load_dataset(ModelSet.train, 64)