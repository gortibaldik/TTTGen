from .model import Encoder
from .layers import DecoderRNNCell
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import tensorflow as tf
import numpy as np
import time
import os


def eval_step( batch_data
             , encoder
             , decoderRNNCell
             , batch_size
             , eval_accuracy_metrics):
    dec_inputs, targets, *tables = batch_data
    enc_outs, *last_hidden_rnn = encoder(tables)
    initial_state = [last_hidden_rnn[-1], *last_hidden_rnn]
    decoderRNNCell.initialize_enc_outs(enc_outs)
    dec_in = dec_inputs[:, 0, :] # start tokens

    result_preds = np.zeros(targets.shape, dtype=np.int)

    for t in range(targets.shape[1]):
        pred, initial_state = decoderRNNCell(dec_in, initial_state)
        predicted_ids = tf.argmax(pred, axis=1).numpy()
        result_preds[:, t] = predicted_ids
        dec_in = tf.expand_dims(predicted_ids, 1)
    
    # gather accurracy after eval step
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    eval_accuracy_metrics.update_state(targets, result_preds, sample_weight=mask)

    return result_preds


def evaluate( dataset
            , steps
            , batch_size
            , max_sum_size
            , ix_to_tk
            , output_dir
            , eos
            , encoder
            , decoderRNNCell):
    start = time.time()
    print(f"-- started evaluation --", flush=True)
    eval_accuracy_metrics = tf.keras.metrics.Accuracy()
    targets_for_bleu = []
    predictions_for_bleu = []
    print("[", end="")
    ten_percent = steps // 10
    for num, batch_data in enumerate(dataset.take(steps)):
        summaries, *tables = batch_data
        sums = tf.expand_dims(summaries, axis=-1)
        batch_data = (sums[:, :max_sum_size, :], summaries[:, 1:max_sum_size+1], *tables)
        result_preds =  eval_step( batch_data
                                  , encoder
                                  , decoderRNNCell
                                  , batch_size
                                  , eval_accuracy_metrics)
        for rp, tgt in zip(result_preds, summaries[:, 1:max_sum_size+1]):
            # find the first occurrence of padding
            # one index before it is <<EOS>> token
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
    with open( os.path.join(output_dir, "predicted" + time.strftime("%Y%m%d%H%M%S") + ".txt")
             , 'w') as f:
        for prediction in predictions_for_bleu:
            print(" ".join(prediction), file=f)
    
    print(f"accurracy : {eval_accuracy_metrics.result()} bleu : {bleu}")
    print(f"-- evaluation duration : {time.time() - start}")
