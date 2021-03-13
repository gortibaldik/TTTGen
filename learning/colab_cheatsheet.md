# Access to google drive through colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

# How to keep your sanity with colab saving policies
- set up access to google drive
- save the dataset to drive and load it from drive
- set up checkpoints (a policy for saving partially completed models) which will send the data to google drive where the data would rest no matter what happens with the colab notebook

### Example of creating checkpoints with `tensorflow`
```python
checkpoint_dir = '/content/drive/My Drive/encoder_decoder_basic/training_checkpoints/'
checkpoint_prefix_m1 = os.path.join(checkpoint_dir, 'ckpt_m1')
# `optimizer` is e.g. `tf.keras.optimizers.Adam`
# `encoder_m1` and `decoder_m1` are my custom models
checkpoint_m1 = tf.train.Checkpoint( optimizer=optimizer
                                   , encoder=encoder_m1
                                   , decoder=decoder_m1)
```

### Example of saving to checkpoints
```python
checkpoint.save(file_prefix=checkpoint_prefix)
```

### Example of loading from checkpoints
```python
checkpoint_m1.restore(tf.train.latest_checkpoint(checkpoint_dir))
```
