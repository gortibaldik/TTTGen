# Experiments with the code provided by various tutorials

- basic encoder with attention based on [tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model) : [jupiter notebook](./bewa_seq2seq.ipynb)

- results and characteristics of the models
- train-test split : `9:1`

### Basic model
- basic encoder-decoder architecture, one layer GRU encoder, one layer GRU decoder, greedy search on inference
- results on the test dataset: `f1 = 0.4665`

### Two-layer basic model
- two layer GRU encoder, two layer GRU decoder, greedy search on inference
- results on the test dataset: `f1 = 0.4506`

### One-layer encoder-decoder with Bahdanau attention
- results on the test dataset : `f1 = 0.561`

### Two-layer encoder-decoder with Bahdanau attention
- results on the test dataset : `f1 = 0.609`
- the __best__ results obtained

### Further possibilities
- beam search on inference
- using other than Bahdanau attention (e.g. Luong)
- since it isn't the main point of interest, no further development of the models is expected
