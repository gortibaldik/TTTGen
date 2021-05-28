# Code for my thesis: Generating text from structured data

## Creation of the Dataset

We train our models on special tensorflow [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format of the datasets. Running the[`create_bpe_dataset.sh`](.preprocessing/create\_bpe\_dataset.sh) script the datasets for the models.

The script accepts three positional command line arguments: number of merges in [Byte Pair Encoding](https://github.com/rsennrich/subword-nmt), directory where to save the dataset and directory with the original dataset. The optional arguments provide following options:


   - `--adv` : lowercase the dataset
   - `--content_plan` : create the dataset with the content plans (for training of the *CS&P* model)
   - `--order_records` : order the records (this makes the structure of the input data easier to understand for the network)
   - `--prun_records` : order the records and erase the unsignificant ones (another method for simplifying the data)
   - `--tfrecord` : save the dataset to `.tfrecord` format
   - `--npy` and `--txt` serve only debugging purposes

## Training of the Models
Having prepared the dataset, we can start training any of the models discussed in the thesis by running [`train.py`](./train.py) script. It accepts numerous command line arguments that are described after calling `train.py --help`.

We also show an example how to train the baseline model discussed in section (assuming the prepared dataset is in directory `ni_tfrecord`:

```shell
python3 train.py --path=ni_tfrecord --batch_size=8 \
    --word_emb_dim=600 --tp_emb_dim=300 --ha_emb_dim=300 \
    --hidden_size=600 --attention_type=dot \
    --decoder_type=baseline --truncation_size=100 \
    --truncation_skip_step=25 --dropout_rate=0.3 \
    --scheduled_sampling_rate=0.8
```

## Generation with Trained Model


After having trained the model we can use [`generate.py`](./generate.py) script to generate summaries of all the tables from the validation and test dataset. It is important to note that even a slight change in the dataset files may result in broken generation, therefore we advise the reader to use exactly the same dataset both for training and for generation.

To allow generating with the models presented in this thesis we publish all the models along with datasets used for their training on [google drive](https://drive.google.com/drive/folders/1eRzAOaZ2SLHOiYm2xtazsb3XSc5ILgbv?usp=sharing) along with exact instructions how to set up the hyperparameters.
