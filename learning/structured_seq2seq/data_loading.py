from enum import Enum
import statistics
import numpy as np
import tensorflow as tf

class ModelSet(Enum):
    DATA_DIR = "processed_data"
    train = [DATA_DIR + '/train/train.summary.id',
             DATA_DIR + '/train/train.box.val.id',
             DATA_DIR + '/train/train.box.lab.id',
             DATA_DIR + '/train/train.box.pos',
             DATA_DIR + '/train/train.box.rpos',
             DATA_DIR + '/train/train.box.val',
             DATA_DIR + '/train/train.box.lab']
    test =  [DATA_DIR + '/test/test.summary.id',
             DATA_DIR + '/test/test.box.val.id',
             DATA_DIR + '/test/test.box.lab.id',
             DATA_DIR + '/test/test.box.pos',
             DATA_DIR + '/test/test.box.rpos',
             DATA_DIR + '/test/test.box.val',
             DATA_DIR + '/test/test.box.lab' ]
    val =   [DATA_DIR + '/valid/valid.summary.id',
             DATA_DIR + '/valid/valid.box.val.id',
             DATA_DIR + '/valid/valid.box.lab.id',
             DATA_DIR + '/valid/valid.box.pos',
             DATA_DIR + '/valid/valid.box.rpos',
             DATA_DIR + '/valid/valid.box.val',
             DATA_DIR + '/valid/valid.box.lab']

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
    data = data.batch(batch_size)
    return data




if __name__ == "__main__":
    load_dataset(ModelSet.train, 64)