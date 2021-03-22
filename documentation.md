# Materials

- how to use tensorflow and pytorch : http://cs231n.stanford.edu/slides/2020/lecture_6.pdf
- implementation of different attention mechanisms in Tensorflow : https://github.com/uzaymacar/attention-mechanisms 
- introduction to NLP with neural nets https://canvas.education.lu.se/courses/3766/pages/chapter-16-natural-language-processing-with-rnns-and-attention?module_item_id=233955
- web titles for web pages : https://arxiv.org/pdf/1807.00099.pdf
- table to text : http://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-Chen.pdf

### What is `tf.function`
- official guide from tensorflow : https://www.tensorflow.org/guide/function#setup
- analysis of `tf.function` : [part1](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/), [part2](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/)
- discussion about the issue I faced (training multiple independent models using the same `tf.function`) [stack_overflow](https://stackoverflow.com/questions/60704587/training-multiple-models-defined-from-the-same-class-in-tensorflow-2-0-fails-whe)

### Using `tf.checkpoint`
- https://www.tensorflow.org/guide/checkpoint#loading_mechanics

### Basic info on RNNs
- introduction to RNNs : https://www.deeplearningbook.org/contents/rnn.html#pf4
- introduction to LSTM : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- implementation of LSTM in pytorch : https://theaisummer.com/understanding-lstm/

### Neural Machine Translation With Attention
- introduction to everything around RNNs used for neural machine translation : https://github.com/lmthang/thesis
- paper about seq to seq architecture without attention : https://arxiv.org/pdf/1409.3215.pdf
- tutorial in new API : https://www.tensorflow.org/tutorials/text/nmt_with_attention
- tutorial introducing `tfa.seq2seq.BeamSearchDecoder` : https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
- tutorial in old API, which is much more verbose and contains many additional materials : https://github.com/tensorflow/nmt#background-on-the-attention-mechanism

### Defining custom RNN cells
- https://www.tensorflow.org/guide/keras/rnn#rnn_layers_and_rnn_cells

### Hyperparameter tuning in Seq2Seq models
- Massive Exploration of Neural Machine Translation Architectures :  https://arxiv.org/pdf/1703.03906.pdf

### existing Seq2Seq neural frameworks
- neural monkey : https://github.com/ufal/neuralmonkey
- tf-seq2seq, which has quite nicely elaborated documentation : https://github.com/google/seq2seq
- deep-shallow, written in pytorch, fairseq : https://github.com/jungokasai/deep-shallow

### Natural language generation - first phase
#### Just gathering information about architectures
- interesting survey about many aspects of language generation : https://arxiv.org/pdf/1703.09902.pdf
- Few-Shot NLG with Pre-Trained Language Model: [paper](https://www.aclweb.org/anthology/2020.acl-main.18.pdf), [code](https://github.com/czyssrs/Few-Shot-NLG)
- Neural Text Generation from Structured Data with Application to the Biography Domain : https://www.aclweb.org/anthology/D16-1128.pdf
- Table-to-text Generation by Structure-aware Seq2seq Learning : [paper](https://arxiv.org/pdf/1711.09724.pdf), [code](https://github.com/tyliupku/wiki2bio), [code in newer tf api](https://github.com/Parth27/Data2Text)

### Natural language generation - second phase
#### Generating sport match summaries - RotoWire dataset
- Data-to-Text Generation with Content Selection and Planning [paper](https://arxiv.org/pdf/1809.00582.pdf), [code](https://github.com/ratishsp/data2text-plan-py)

### Datasets
- the main dataset I'm working with : WIKIBIO dataset : [dataset](https://github.com/DavidGrangier/wikipedia-biography-dataset), [paper introducing it](https://arxiv.org/abs/1603.07771)
- rotowire dataset which would become the main dataset of my thesis: [dataset](https://github.com/harvardnlp/boxscore-data), [paper introducing it](https://arxiv.org/pdf/1707.08052.pdf)
- sportsett basketball dataset which should be the ultimate replacement of the rotowire dataset according to the authors of the rotowire dataset: [dataset](https://github.com/nlgcat/sport_sett_basketball), [paper introducing it](https://intellang.github.io/papers/5-IntelLanG_2020_paper_5.pdf)
