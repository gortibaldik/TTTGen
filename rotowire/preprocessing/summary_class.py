try:
    from constants import number_words, name_transformations # pylint: disable=import-error
    from utils import join_strings, OccurrenceDict # pylint: disable=import-error
except:
    from .constants import number_words, name_transformations
    from .utils import join_strings, OccurrenceDict
from text_to_num import text2num
import nltk.tokenize as nltk_tok

class Summary:
    """Summary holds the summary part of the datapoint present in the RotoWire dataset.
    
    The class provides numerous preprocessing methods"""
    @staticmethod
    def traverse_span(span, entities_set):
        """
        traverse span of word tokens and concatenate it until the actual concatenation 
        of words isn't part of the entities_set
        :return: entity found in the span and number of words in the entity
        """
        candidate = span[0]
        index = 1
        while index < len(span) and join_strings(candidate, span[index]) in entities_set:
            candidate = join_strings(candidate, span[index])
            index += 1
        return index, candidate

    @staticmethod
    def extract_entities(sentence, entities_set):
        """
        Traverse the sentence and try to extract all the
        named entities present in it
        - problem: all the substrings present in the span must be in the entities_set, therefore
        if we search for Luc Mbah a Moute then {"Luc", "Luc Mbah", "Luc Mbah a", "Luc Mbah a Moute"} must
        be a subset of the entities set
        :return: list with all the extracted named entities
        """
        index = 0
        tokenized_sentence = sentence.split()
        candidates = []
        while index < len(tokenized_sentence):
            if tokenized_sentence[index] in entities_set:
                i, candidate = Summary.traverse_span(tokenized_sentence[index:], entities_set)
                index += i
                candidates.append(candidate)
            else:
                index += 1

        return candidates

    @staticmethod
    def transform_numbers(sent):
        """Traverse the tokens and collect all the number words and transform them into a numeral"""
        def has_to_be_ignored(__sent, __i):
            ignores = { 
              "three point",
              "three - point",
              "three - pt",
              "three pt",
              "three - pointers",
              "three - pointer",
               "three pointers"
            }
            return " ".join(__sent[__i:__i + 3]) in ignores or " ".join(__sent[__i:__i + 2]) in ignores

        def extract_number_literal(word):
            """
            Detects literals like "22"
            """
            try:
                __number = int(word)
                return True, __number
            except ValueError:
                return False, None

        def extract_number_words(span):
            """
            Detects literals like "twenty two"
            """
            index = 0
            while index < len(span) and span[index] in number_words and not has_to_be_ignored(span, index):
                index += 1
            if index == 0:
                return 1, span[0]
            try:
                __result = text2num(" ".join(span[:index]), lang="en")
                return index, __result
            except ValueError:
                return index, " ".join(span[:index])

        extracted_sentence = []
        i = 0
        sent = sent.split()
        while i < len(sent):
            token = sent[i]
            is_number_literal, number = extract_number_literal(token)
            if is_number_literal:
                extracted_sentence.append(str(number))
                i += 1
            elif token in number_words and not has_to_be_ignored(sent, i):
                j, result = extract_number_words(sent[i:])
                extracted_sentence.append(str(result))
                i += j
            else:
                extracted_sentence.append(token)
                i += 1
        return " ".join(extracted_sentence)

    def collect_tokens(self, word_dict : OccurrenceDict):
        """ Add all the tokens from the summary to the word_dict"""
        for token in self._list_of_words:
            word_dict.add(token)

    @staticmethod
    def _transform_words(list_of_words, words_limit=None):
        """ Traverse through the summary and transform dataset faults
        
        E.g. we transform Barea’s to Barea ’s, all the version of name Luc Mbah A Moute to Moute, all the number
        words to numerals etc. """
        summary = join_strings(*list_of_words)
        sentences = [Summary.transform_numbers(s) for s in nltk_tok.sent_tokenize(summary)]
        result = []
        for s in sentences:
            tokens = []
            # transform possessives
            for token in s.strip().split():
                if token.endswith('’s'):
                    tokens.append(token.replace('’s', ''))
                    tokens.append("’s")
                else:
                    tokens.append(token)
            ix = 0
            candidate_sentence = []
            # transform dataset faults
            while ix < len(tokens):
                found = False
                for r in range(5, 0, -1):
                    multi_tokens = " ".join(tokens[ix:ix+r])
                    if multi_tokens in name_transformations:
                        candidate_sentence += name_transformations[multi_tokens]
                        found = True
                        ix += r
                        break

                if not found:
                    candidate_sentence.append(tokens[ix])
                    ix += 1
            if (words_limit is not None) and (len(result) + len(candidate_sentence) > words_limit):
                break
            else:
                result += candidate_sentence

        return result

    def get_entities_from_summary(self, entities_set):
        """
        Traverse the summary and try to extract all the named entities present in it
        - problem: all the substrings present in the summary must be in the entities_set, therefore
        if we search for "Stephen Curry" both "Stephen" and "Stephen Curry" must be present in the
        entities_set
        -----
        :return: list with all the extracted named entities
        """
        summary = join_strings(*self._list_of_words)
        extracted = []
        for s in nltk_tok.sent_tokenize(summary):
            extracted += self.extract_entities(s, entities_set)
        return extracted

    def transform( self
                 , transformations
                 , lowercase=False):
        """Traverse the summary and transform the longest subsequences of words present in the transformations dict"""
        new_list_of_words = []
        ix = 0
        length = len(self._list_of_words)
        while ix < length:
            found = False
            for r in range(3, 0, -1):
                candidate = " ".join(self._list_of_words[ix:ix+r])
                if candidate in transformations:
                    ix += r
                    if transformations[candidate] != "":
                        new_list_of_words.append(transformations[candidate])
                    found = True
                    break
            if not found:
                new_list_of_words.append(self._list_of_words[ix].lower() if lowercase
                                         else self._list_of_words[ix])
                ix += 1
        self._list_of_words = new_list_of_words

    def get_words(self):
        return self._list_of_words

    def __init__( self
                , list_of_words
                , word_dict
                , words_limit=None):
        """ Initialize the summary, remove all the dataset faults, transform number words to numerals and collect tokens to word_dict"""
        self._list_of_words = self._transform_words(list_of_words, words_limit=words_limit)
        self.collect_tokens(word_dict)

    def __str__(self):
        return " ".join(self._list_of_words)

    def __len__(self):
        return self._list_of_words.__len__()