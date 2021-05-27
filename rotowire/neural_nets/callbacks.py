import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np

class CalcBLEUCallback(tf.keras.callbacks.Callback):
    """ Callback which calculates BLEU score of the summaries generated from the dataset.

    Callback gets called at the end of each epoch
    """
    def __init__(self, dataset, ix_to_tk, eos):
        """ Initialize CalcBLEUCallback

        process the dataset, save ix_to_tk, and index of the eos token

        Args:
            dataset:    dataset which will be used for generation of the summaries
                        and calculation of the BLEU score
            ix_to_tk:   vocabulary for transforming generated sequences of indices to words
            eos:        index of the end of sequence token
        """
        super(CalcBLEUCallback, self).__init__()
        targets = None
        for tgt in dataset.as_numpy_iterator():
            summaries, *_ = tgt
            if targets is None:
                targets = summaries[:, 1:]
            else:
                targets = np.append(targets, summaries[:, 1:], axis=0)
        self._targets = targets
        self.dataset = dataset
        self._ix_to_tk = ix_to_tk
        self._eos = eos

    def on_epoch_end(self, epoch, logs=None):
        """ Calculate BLEU at the end of the epoch"""
        print("Predicting at the end of epoch")
        predictions = self.model.predict(self.dataset)
        print(predictions.shape)
        print("Creating output sentences with vocab")
        predictions_for_bleu = []
        targets_for_bleu = []
        for rp, tgt in zip(predictions, self._targets):
            # find the first occurrence of <<EOS>> token
            tpl = np.nonzero(tgt == self._eos)[0]
            ix = tpl[0] if len(tpl) > 0 else tgt.shape[0]
            targets_for_bleu.append([[self._ix_to_tk[i] for i in tgt[:ix]]])
            tpl = np.nonzero(rp == self._eos)[0]
            ix = tpl[0] if len(tpl) > 0 else ix
            predictions_for_bleu.append([self._ix_to_tk[i] for i in rp[:ix]])
        
        print(" ".join(predictions_for_bleu[5]))
        print(" ".join(targets_for_bleu[5][0]))
        print("calculating bleu")
        bleu = corpus_bleu( targets_for_bleu
                          , predictions_for_bleu
                          , smoothing_function=SmoothingFunction().method4)
        print(f"BLEU : {bleu}")

class SaveOnlyModelCallback(tf.keras.callbacks.Callback):
    """ Our take on the tf.keras.callbacks.ModelCheckpoint which saves model at the end of each epoch """
    def __init__(self, checkpoint, checkpoint_prefix):
        """
        Args:
            checkpoint: tf.train.Checkpoint which manages the actual model
            checkpoint_prefix: file_prefix argument of checkpoint.save()
        """
        super(SaveOnlyModelCallback, self).__init__()
        self._checkpoint = checkpoint
        self._checkpoint_prefix = checkpoint_prefix

    def on_epoch_end(self, epoch, logs=None):
        """ Save the model at the end of each epoch"""
        print(self._checkpoint.save(file_prefix=self._checkpoint_prefix))