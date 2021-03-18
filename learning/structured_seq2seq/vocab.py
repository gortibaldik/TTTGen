class Vocab(object):
    """vocabulary for words and field types"""
    PAD = 'PAD'
    START_TOKEN = 'START_TOKEN'
    END_TOKEN = 'END_TOKEN'
    UNK_TOKEN = 'UNK_TOKEN'

    def __init__(self, field_vocab_path, word_vocab_path):
        vocab = dict()
        vocab[Vocab.PAD] = 0
        vocab[Vocab.START_TOKEN] = 1
        vocab[Vocab.END_TOKEN] = 2
        vocab[Vocab.UNK_TOKEN] = 3
        cnt = 4
        with open(word_vocab_path, "r", encoding='utf8') as v:
            for line in v:
                word = line.strip().split()[0]
                vocab[word] = cnt
                cnt += 1
        self._word2id = vocab
        self._id2word = {value: key for key, value in vocab.items()}

        key_map = dict()
        key_map[Vocab.PAD] = 0
        key_map[Vocab.START_TOKEN] = 1
        key_map[Vocab.END_TOKEN] = 2
        key_map[Vocab.UNK_TOKEN] = 3
        cnt = 4
        with open(field_vocab_path, "r", encoding='utf8') as v:
            for line in v:
                key = line.strip().split()[0]
                key_map[key] = cnt
                cnt += 1
        self._key2id = key_map
        self._id2key = {value: key for key, value in key_map.items()}

    def get_words_size(self):
        return len(self._word2id)

    def get_fields_size(self):
        return len(self._key2id)

    def ids2words(self, ids):
        result = []
        for id in ids:
            result.append(self.id2word(id))
        return result

    def word2id(self, word):
        ans = self._word2id[word] if word in self._word2id else 3
        return ans

    def id2word(self, id):
        ans = self._id2word[int(id)]
        return ans

    def key2id(self, key):
        ans = self._key2id[key] if key in self._key2id else 3
        return ans

    def id2key(self, id):
        ans = self._id2key[int(id)]
        return ans