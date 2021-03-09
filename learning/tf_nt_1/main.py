import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import unicodedata
import re
import numpy as np
import os
import io
import time


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    source, target, _ = zip(*word_pairs)
    return source, target


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, n_layers, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.recurrences = []
        for _ in range(n_layers):
            self.recurrences.append(tf.keras.layers.GRU(self.enc_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        recurrent_initializer='glorot_uniform'))

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden_state = [None] * len(self.recurrences)
        output = x
        for l in range(len(self.recurrences)):
            output, hidden_state[l] = self.recurrences[l](output, initial_state=hidden[l])
        return output, hidden_state

    def initialize_hidden_state(self):
        hidden_state = []
        for _ in range(len(self.recurrences)):
            hidden_state.append(tf.zeros((self.batch_sz, self.enc_units)))
        return hidden_state


class DecoderWA(tf.keras.Model):
    """ Decoder without attention
        - computes the predictions from hidden encoder state
    """
    def __init__(self, vocab_size, embedding_dim, n_layers, dec_units, batch_sz):
        super(DecoderWA, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.recurrences = []
        for _ in range(n_layers):
            self.recurrences.append(tf.keras.layers.GRU(self.dec_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        recurrent_initializer='glorot_uniform'))
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        # since this is Decoder Without Attention
        # enc_output sequence isn't used here
        x = self.embedding(x)
        hidden_state = [None] * len(self.recurrences)
        output = x
        for l in range(len(self.recurrences)):
            output, hidden_state[l] = self.recurrences[l](output, initial_state=hidden[l])
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, hidden_state, None


def loss_function(x, y, loss_object):
    mask = tf.math.logical_not(tf.math.equal(y, 0))
    loss_ = loss_object(y, x)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


class TrainStepWrapper:
    @tf.function
    def train_step( self
                  , inp
                  , targ
                  , enc_hidden
                  , targ_lang
                  , encoder
                  , decoder
                  , loss_object
                  , optimizer
                  , BATCH_SIZE):
        """
        :type encoder: Encoder
        :type decoder: DecoderWA
        """
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function( predictions
                                     , targ[:, t]
                                     , loss_object)
                # teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = loss / int(targ.shape[1])
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


def train( dataset
         , encoder
         , decoder
         , loss_object
         , optimizer
         , n_epochs
         , batch_size
         , steps_per_epoch
         , targ_lang
         , checkpoint
         , checkpoint_prefix):
    tsw = TrainStepWrapper()
    for epoch in range(n_epochs):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = tsw.train_step( inp
                                       , targ
                                       , enc_hidden
                                       , targ_lang
                                       , encoder
                                       , decoder
                                       , loss_object
                                       , optimizer
                                       , batch_size)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time() - start} sec\n')


def evaluate( inp
            , targ
            , encoder
            , decoder
            , inp_lang
            , targ_lang
            , max_length_inp
            , max_length_targ
            , units
            , n_layers):
    BATCH_SIZE = inp.shape[0]
    enc_hidden = []
    for _ in range(n_layers):
        enc_hidden.append(tf.zeros((BATCH_SIZE, units)))
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
    result_preds = np.zeros(targ.shape)

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        predicted_ids = tf.argmax(predictions, axis=1).numpy()
        result_preds[:, t] = predicted_ids
        dec_input = tf.expand_dims(predicted_ids, 1)
    f1 = 0
    end_index = targ_lang.word_index['<end>']
    for s in range(targ.shape[0]):
        indices = np.where(result_preds[s] == end_index)
        index = np.where(targ[s] == end_index)[0][0]
        if len(indices[0]) != 0:
            index = np.maximum(index, indices[0][0])
        predicted = result_preds[s, 1:index+1].astype(np.int)
        expected = targ[s, 1:index+1].astype(np.int)
        f1 += f1_score(expected, predicted, average='macro')

    print(f"Average f1 score over validation dataset : {f1 / targ.shape[0]}")


def translate( sentence
             , encoder
             , decoder
             , inp_lang
             , targ_lang
             , max_length_inp
             , max_length_targ
             , units):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
             [inputs],
             maxlen=max_length_inp,
             padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden = decoder( dec_input
                                         , dec_hidden
                                         , enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = targ_lang.index_word[predicted_id]
        result += '' if t == 0 else ' '
        result += predicted_word

        if predicted_word == '<end>':
            print(f'Input: {sentence}')
            print(f'Predicted translation: {result}')
            return

        dec_input = tf.expand_dims([predicted_id], 0)

    print(f'Input: {sentence}')
    print(f'Predicted translation: {result}')


def _main():
    path = "spa.txt"
    num_examples = 30000
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path, num_examples)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    # Show length
    print(f"training examples: {len(input_tensor_train)}")
    print(f"testing examples: {len(input_tensor_val)}")

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    n_layers = 1
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    # this creates a training dataset
    train_data = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)

    # create a test dataset
    test_data = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))

    example_input_batch, example_target_batch = next(iter(train_data))
    print(f"batch shape: {example_input_batch.shape}")

    encoder = Encoder(vocab_inp_size, embedding_dim, n_layers, units, BATCH_SIZE)

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print(f'Encoder output shape: (batch size, sequence length, units) {sample_output.shape}')
    print(f'Encoder Hidden state shape: (batch size, units) {sample_hidden[0].shape}')

    decoder = DecoderWA(vocab_tar_size, embedding_dim, n_layers, units, BATCH_SIZE)
    sample_decoder_output, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          sample_hidden, sample_output)

    print(f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, reduction='none')

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint( optimizer=optimizer
                                    , encoder=encoder
                                    , decoder=decoder)

    evaluate(input_tensor_val
             , target_tensor_val
             , encoder
             , decoder
             , inp_lang
             , targ_lang
             , max_length_inp
             , max_length_targ
             , units)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_data.take(steps_per_epoch)):
            batch_loss = train_step( inp
                                   , targ
                                   , enc_hidden
                                   , targ_lang
                                   , encoder
                                   , decoder
                                   , loss_object
                                   , optimizer
                                   , BATCH_SIZE)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time() - start} sec\n')

    translate(u'hace mucho frio aqui.'
             , encoder
             , decoder
             , inp_lang
             , targ_lang
             , max_length_inp
             , max_length_targ
             , units)


if __name__ == "__main__":
    _main()