# Materials

### Introductory info on RNNs
- introduction to RNNs : https://www.deeplearningbook.org/contents/rnn.html#pf4
- introduction to LSTM : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- implementation of LSTM in pytorch : https://theaisummer.com/understanding-lstm/

### Introduction to NLP and `tensorflow`
- how to use tensorflow and pytorch : http://cs231n.stanford.edu/slides/2020/lecture_6.pdf
- introduction to NLP with neural nets https://canvas.education.lu.se/courses/3766/pages/chapter-16-natural-language-processing-with-rnns-and-attention?module_item_id=233955

### Neural Machine Translation With Attention
- implementation of different attention mechanisms in Tensorflow : https://github.com/uzaymacar/attention-mechanisms 
- introduction to everything around RNNs used for neural machine translation : https://github.com/lmthang/thesis
- paper about seq to seq architecture without attention : https://arxiv.org/pdf/1409.3215.pdf
- tutorial in new API : https://www.tensorflow.org/tutorials/text/nmt_with_attention
- tutorial introducing `tfa.seq2seq.BeamSearchDecoder` : https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
- tutorial in old API, which is much more verbose and contains many additional materials : https://github.com/tensorflow/nmt#background-on-the-attention-mechanism

### Hyperparameter tuning in Seq2Seq models
- Massive Exploration of Neural Machine Translation Architectures :  https://arxiv.org/pdf/1703.03906.pdf
- broad tutorial on seq2seq architectures : https://arxiv.org/pdf/1703.01619.pdf
- Byte Pair Encoding: [blog of Lei Mao](https://leimao.github.io/blog/Byte-Pair-Encoding/), [paper (Sennrich et al.)](https://www.aclweb.org/anthology/P16-1162/), [code released by Sennrich et al.](https://github.com/rsennrich/subword-nmt)
- teacher forcing, implementation of attention mechanisms in pytorch: [code in pytorch](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation), [paper about scheduled sampling](https://arxiv.org/pdf/1506.03099.pdf)
- Neural Machine Translation with Reconstruction: [paper](https://arxiv.org/pdf/1611.01874.pdf)
- What is perplexity : https://planspace.org/2013/09/23/perplexity-what-it-is-and-what-yours-is/
- Regularisation : https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize
- Truncated BackPropagation through time : https://r2rt.com/styles-of-truncated-backpropagation.html

### existing Seq2Seq neural frameworks
- openNMT, looks like the best solution using tensorflow : https://github.com/OpenNMT/OpenNMT-tf
- neural monkey : https://github.com/ufal/neuralmonkey
- tf-seq2seq, which has quite nicely elaborated documentation : https://github.com/google/seq2seq
- deep-shallow, written in pytorch, fairseq : https://github.com/jungokasai/deep-shallow

--------------

### Natural language generation - first phase
#### Just gathering information about architectures
- interesting survey about many aspects of language generation : https://arxiv.org/pdf/1703.09902.pdf
- Few-Shot NLG with Pre-Trained Language Model: [paper](https://www.aclweb.org/anthology/2020.acl-main.18.pdf), [code](https://github.com/czyssrs/Few-Shot-NLG)
- Neural Text Generation from Structured Data with Application to the Biography Domain : https://www.aclweb.org/anthology/D16-1128.pdf
- Table-to-text Generation by Structure-aware Seq2seq Learning : [paper](https://arxiv.org/pdf/1711.09724.pdf), [code](https://github.com/tyliupku/wiki2bio), [code in newer tf api](https://github.com/Parth27/Data2Text)
- table to text : http://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-Chen.pdf
- copyNet code : https://github.com/lspvic/CopyNet/blob/master/copynet.py
- copyNet paper : https://arxiv.org/abs/1603.06393
- reference-aware language models : https://arxiv.org/pdf/1611.01628.pdf
- SO on negative sampling : [SO](https://stackoverflow.com/questions/37671974/tensorflow-negative-sampling), [tf](https://www.tensorflow.org/extras/candidate_sampling.pdf)

------------------

### Natural language generation - second phase
#### Generating sport match summaries - RotoWire dataset
- baselines from the authors of the dataset : [code](https://github.com/harvardnlp/data2text)
- Data-to-Text Generation with Content Selection and Planning [paper](https://arxiv.org/pdf/1809.00582.pdf), [code](https://github.com/ratishsp/data2text-plan-py)
- Information Extraction from Rotowire summaries [code](https://github.com/ratishsp/data2text-1) 
- A Hierarchical Model for Data-to-Text Generation [paper](https://arxiv.org/pdf/1912.10011v1.pdf), [code](https://github.com/KaijuML/data-to-text-hierarchical)
- Reference aware language models - how to encode individual dataset record into a representation understandable by neural net [paper](https://arxiv.org/abs/1611.01628)
- Rotowire isn't good dataset : [paper](https://www.aclweb.org/anthology/W19-8639.pdf)

------------------

### Datasets
- the main dataset I'm working with : WIKIBIO dataset : [dataset](https://github.com/DavidGrangier/wikipedia-biography-dataset), [paper introducing it](https://arxiv.org/abs/1603.07771)
- rotowire dataset which would become the main dataset of my thesis: [dataset](https://github.com/harvardnlp/boxscore-data), [paper introducing it](https://arxiv.org/pdf/1707.08052.pdf)
- sportsett basketball dataset which should be the ultimate replacement of the rotowire dataset according to the authors of the rotowire dataset: [dataset](https://github.com/nlgcat/sport_sett_basketball), [paper introducing it](https://intellang.github.io/papers/5-IntelLanG_2020_paper_5.pdf)

----------------

### Used `tensorflow` tutorials
- [`checkpoints`](https://www.tensorflow.org/guide/checkpoint#loading_mechanics)
- [`tf.function` tutorial from tensorflow](https://www.tensorflow.org/guide/function#setup)
- analysis of `tf.function` : [part1](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/), [part2](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/)
- discussion about the issue I faced (training multiple independent models using the same `tf.function`) [stack_overflow](https://stackoverflow.com/questions/60704587/training-multiple-models-defined-from-the-same-class-in-tensorflow-2-0-fails-whe)
- [custom RNN cells](https://www.tensorflow.org/guide/keras/rnn#rnn_layers_and_rnn_cells)
- [tf keras metrics](https://neptune.ai/blog/keras-metrics)
- [custom training loop](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [customizing what happens in fit()](https://keras.io/guides/customizing_what_happens_in_fit/)
- [slicing tensor in graph mode](https://www.tensorflow.org/guide/tensor_slicing)

--------------

### `tensorflow` performance analysis
- forwarding ports from cluster to localhost: [stack overflow](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)
- `tensorflow` guide on profiling custom training loops: https://www.tensorflow.org/guide/profiler

### `latex` resources
- [text highlighting](https://texblog.org/2015/05/20/using-colors-in-a-latex-document/)
