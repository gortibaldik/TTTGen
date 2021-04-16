from .model import Encoder
from .layers import DecoderRNNCell, DecoderRNNCellJointCopy
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import tensorflow as tf
import numpy as np
import time
import os

def eval_step( batch_data
             , encoder
             , decoderRNNCell
             , batch_size
             , eval_accuracy_metrics
             , eval_scc_metrics
             , generate : bool = False):
    dec_inputs, targets, *tables = batch_data
    enc_outs, *last_hidden_rnn = encoder(tables)

    if isinstance(decoderRNNCell, DecoderRNNCellJointCopy):
        enc_ins = tf.one_hot(tf.cast(tables[2], tf.int32), decoderRNNCell._word_vocab_size)
        aux_inputs = (enc_outs, enc_ins) # value portion of the record needs to be copied
    else:
        aux_inputs = (enc_outs,)

    initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]
    dec_in = dec_inputs[:, 0, :] # start tokens

    result_preds = np.zeros(targets.shape, dtype=np.int)

    for t in range(targets.shape[1]):
        pred, initial_state = decoderRNNCell( (dec_in, *aux_inputs)
                                            , initial_state
                                            , training=False)
        if not generate:
            eval_scc_metrics.update_state(targets[:, t], pred, sample_weight=tf.math.logical_not(tf.math.equal(targets[:, t], 0)) )
        predicted_ids = tf.argmax(pred, axis=1).numpy()
        result_preds[:, t] = predicted_ids
        if generate:
            dec_in = tf.expand_dims(predicted_ids, 1)
        else: # use teacher forcing, this will measure the perplexity of network on validation dataset
            dec_in = tf.expand_dims(targets[:, t], axis=1)
    
    # gather accurracy after eval step if not generating
    if not generate:
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        eval_accuracy_metrics.update_state(targets, result_preds, sample_weight=mask)
        

    return result_preds


def evaluate( dataset
            , steps
            , batch_size
            , ix_to_tk
            , output_dir
            , eos
            , encoder
            , decoderRNNCell):
    start = time.time()
    print(f"-- started evaluation --", flush=True)
    eval_accuracy_metrics = tf.keras.metrics.Accuracy()
    eval_scc_metrics = tf.keras.metrics.SparseCategoricalCrossentropy()
    targets_for_bleu = []
    predictions_for_bleu = []
    print("[", end="")
    ten_percent = steps // 10
    bleu_batches = 43
    for num, batch_data in enumerate(dataset.take(steps)):
        summaries, *tables = batch_data
        sums = tf.expand_dims(summaries, axis=-1)
        max_sum_size = summaries.shape[1] - 1
        batch_data = (sums[:, :max_sum_size, :], summaries[:, 1:max_sum_size+1], *tables)
        generate = False
        if num % bleu_batches == 0:
            generate = True
        result_preds =  eval_step( batch_data
                                 , encoder
                                 , decoderRNNCell
                                 , batch_size
                                 , eval_accuracy_metrics
                                 , eval_scc_metrics
                                 , generate=generate)

        if num % bleu_batches == 0:
            for rp, tgt in zip(result_preds, summaries[:, 1:max_sum_size+1]):
                # find the first occurrence of <<EOS>> token
                tpl = np.nonzero(tgt == eos)[0]
                ix = tpl[0] if len(tpl) > 0 else max_sum_size
                targets_for_bleu.append([ix_to_tk[i] for i in tgt.numpy()[:ix]])
                tpl = np.nonzero(rp == eos)[0]
                ix = tpl[0] if len(tpl) > 0 else ix
                predictions_for_bleu.append([ix_to_tk[i] for i in rp[:ix]])
        if num % ten_percent == 0:
            print("=", end="")
    print("]")
        
    bleu = corpus_bleu( targets_for_bleu
                      , predictions_for_bleu
                      , smoothing_function=SmoothingFunction().method4)
    # save the predictions
    t = time.strftime("%Y%m%d%H%M%S")
    predictions_path = os.path.join(output_dir, "predicted" + t + ".txt")
    targets_path = os.path.join(output_dir, "gold" + t + ".txt")
    def save_arr(path, arr):
        with open( path
                , 'w'
                , encoding='utf8') as f:
            for line_tokens in arr:
                print(" ".join(line_tokens), file=f)
    
    save_arr(predictions_path, predictions_for_bleu)
    save_arr(targets_path, targets_for_bleu)
        
    final_val_loss = eval_scc_metrics.result()
    print(f"accurracy : {eval_accuracy_metrics.result()} loss : {final_val_loss} bleu : {bleu}")
    print(f"-- evaluation duration : {time.time() - start}")
    return final_val_loss
