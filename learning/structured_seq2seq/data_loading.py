from enum import Enum
import statistics
import numpy as np
import tensorflow as tf

'''
MODEL_SET_DATA_DIR needs to be defined before first use of Enum ModelSet,
then the valid data could be found, otherwise the behavior is undefined
'''
MODEL_SET_DATA_DIR = "/content/drive/My Drive/sequence_aware"

class ModelSet(Enum):
    DATA_DIR = "/content/drive/My Drive/sequence_aware"
    train = [MODEL_SET_DATA_DIR + '/train/train.summary.id',
             MODEL_SET_DATA_DIR + '/train/train.box.val.id',
             MODEL_SET_DATA_DIR + '/train/train.box.lab.id',
             MODEL_SET_DATA_DIR + '/train/train.box.pos',
             MODEL_SET_DATA_DIR + '/train/train.box.rpos',
             MODEL_SET_DATA_DIR + '/train/train.box.val',
             MODEL_SET_DATA_DIR + '/train/train.box.lab']
    test =  [MODEL_SET_DATA_DIR + '/test/test.summary.id',
             MODEL_SET_DATA_DIR + '/test/test.box.val.id',
             MODEL_SET_DATA_DIR + '/test/test.box.lab.id',
             MODEL_SET_DATA_DIR + '/test/test.box.pos',
             MODEL_SET_DATA_DIR + '/test/test.box.rpos',
             MODEL_SET_DATA_DIR + '/test/test.box.val',
             MODEL_SET_DATA_DIR + '/test/test.box.lab' ]
    val =   [MODEL_SET_DATA_DIR + '/valid/valid.summary.id',
             MODEL_SET_DATA_DIR + '/valid/valid.box.val.id',
             MODEL_SET_DATA_DIR + '/valid/valid.box.lab.id',
             MODEL_SET_DATA_DIR + '/valid/valid.box.pos',
             MODEL_SET_DATA_DIR + '/valid/valid.box.rpos',
             MODEL_SET_DATA_DIR + '/valid/valid.box.val',
             MODEL_SET_DATA_DIR + '/valid/valid.box.lab']


def load_dataset(model_set, batch_size):
    summary_path, text_path, field_path, pos_path, rpos_path, vals_path, labs_path = model_set.value
    summaries = np.load(summary_path)
    texts = np.load(text_path)
    fields = np.load(field_path)
    poses = np.load(pos_path)
    rposes = np.load(rpos_path)
    for d in [summaries, texts, fields, poses, rposes]:
        print(d.shape)
    BUFFER_SIZE = len(summaries)

    data = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(summaries),
         tf.convert_to_tensor(texts),
         tf.convert_to_tensor(fields),
         tf.convert_to_tensor(poses),
         tf.convert_to_tensor(rposes))
    ).shuffle(BUFFER_SIZE)
    data = data.batch(batch_size, drop_remainder=True)
    return data


if __name__ == "__main__":
    load_dataset(ModelSet.train, 64)