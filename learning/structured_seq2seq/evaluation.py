from data_loading import ModelSet
from vocab import Vocab
import data_loading
import os
import tensorflow as tf
import numpy as np
import time
from nltk.translate.bleu_score import corpus_bleu


def write_predictions(path, predictions_array):
    with open(path, 'w') as f:
        for prediction in predictions_array:
            f.write(prediction + "\n")


def evaluate( dataset : tf.data.Dataset
            , model_set : ModelSet
            , steps_per_epoch
            , batch_size
            , vocab : Vocab
            , gold_path_prefix : str
            , predictions_path
            , predictions_with_attention_copy_path
            , encoder
            , decoder):
    BATCH_SIZE = batch_size
    count = 0
    vocab_index_start = vocab.word2id(Vocab.START_TOKEN)
    vocab_index_end = vocab.word2id(Vocab.END_TOKEN)
    vocab_index_unk = vocab.word2id(Vocab.UNK_TOKEN)
    predicted_summaries = []
    predicted_summaries_copy_attention = []
    predictions_for_bleu = []

    # to tackle Out Of Vocabulary (OOV) problem
    # the attention is used to copy the relevant data from
    # the original table

    # load the original table :
    ot_path = os.path.join(data_loading.MODEL_SET_DATA_DIR, model_set.value[5])
    original_table = open(ot_path, 'r', encoding='utf8').read().strip().split('\n')
    original_table = [list(t.strip().split(' ')) for t in original_table]

    print("---- Computing predictions")
    start = time.time()
    start_batch = time.time()

    for (num, batch_data) in enumerate(dataset.take(steps_per_epoch)):
        summaries, tables, fields, pos, rpos = batch_data
        outputs, (h, c), field_pos_embeddings = encoder((tables, fields, pos, rpos))
        decoder.initialize_batch(outputs, field_pos_embeddings, h, c)
        dec_input = tf.expand_dims([vocab_index_start] * BATCH_SIZE, 1)

        # go over all the time steps
        result_preds = np.zeros(summaries.shape, dtype=np.int)
        attention_preds = np.zeros(summaries.shape, dtype=np.int)
        for t in range(summaries.shape[1]):
            predictions, attention_vector = decoder(dec_input)

            # generating in greedy manner
            predicted_ids = tf.argmax(predictions, axis=1).numpy()
            result_preds[:, t] = predicted_ids

            # if the generated token is UNK_TOKEN, then
            # the attention is used, to copy the relevant
            # data from the original table
            # save the index of relevant data
            mask = tf.transpose(tf.math.logical_not(tf.math.equal(tables, 0)), [1,0])
            mask = tf.cast(mask, dtype=attention_vector.dtype)
            pred_att = tf.argmax(attention_vector * mask, axis=0).numpy()
            attention_preds[:, t] = pred_att.astype(np.int)

            # feeding last generated index to decoder
            dec_input = tf.expand_dims(predicted_ids, 1)

        # generate the resulting sentences
        for s in range(result_preds.shape[0]):
            indices = np.where(result_preds[s] == vocab_index_end)
            index = np.where(summaries[s] == vocab_index_end)[0][0]
            if len(indices[0]) != 0:
                index = np.maximum(index, indices[0][0])

            predicted = result_preds[s, 1:index + 1].astype(np.int)
            predicted_summaries.append(
                " ".join([vocab.id2word(psx) for psx in predicted])
            )
            predicted_with_attention = []
            for ix, word_ix in enumerate(predicted):
                # substitute the unknown token with the best
                # bet from attention
                if (word_ix == vocab_index_unk) and \
                        (len(original_table[len(predicted_summaries_copy_attention)]) > attention_preds[s, ix]):

                    sub = original_table[len(predicted_summaries_copy_attention)][attention_preds[s, ix]]
                    predicted_with_attention.append(sub)
                else:
                    predicted_with_attention.append(vocab.id2word(word_ix))
            predicted_summaries_copy_attention.append(
                " ".join(predicted_with_attention)
            )
            predictions_for_bleu.append(predicted_with_attention)

        if num % 100 == 0:
            old_start = start_batch
            start_batch = time.time()
            print(f"---- Batch {num} completed in {start_batch - old_start}")

    print(f"---- Net generation completed in {time.time() - start}", flush=True)
    write_predictions(predictions_path, predicted_summaries)
    write_predictions(predictions_with_attention_copy_path, predicted_summaries_copy_attention)

    gold_set = []
    for ix in range(len(predictions_for_bleu)):
        with open(gold_path_prefix + str(ix), 'r', encoding='utf8') as f:
            gold_set.append([f.read().strip().split(' ')])

    print("---- Evaluating", flush=True)
    bleu = corpus_bleu(gold_set, predictions_for_bleu)
    print(f"---- BLEU score : {bleu}", flush=True)