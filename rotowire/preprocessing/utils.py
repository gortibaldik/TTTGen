from collections import defaultdict
from functools import total_ordering

try:
    from constants import BoxScoreEntries, LineScoreEntries, name_transformations # pylint: disable=import-error
except:
    from .constants import BoxScoreEntries, LineScoreEntries, name_transformations
import sys


class EnumDict:
    """
    Dictionary wrapper allowing to index
    by string enums
    """
    def __init__( self, dict):
        self._dict = dict

    def __getitem__(self, item):
        return self._dict[item.value]

    def __setitem__(self, key, value):
        self._dict[key.value] = value

    def __iter__(self):
        return iter(self._dict)

    def map(self, key, fn):
        """ Apply function fn on entry in the internal dictionary indexed by key """
        self[key] = fn(self[key])

    def mapmap(self, key1, key2, fn):
        """ Apply function fn on entry indexed by key2 in dictionary indexed by key1 """
        self[key1][key2] = fn(self[key1][key2])

    def pop(self, key):
        if key.value in self._dict:
            return self._dict.pop(key.value)
        return None 

    def keys(self):
        return self._dict.keys()


class OccurrenceDict:
    """ Dictionary wrapper that collects counts of its entries """
    @total_ordering
    class Unit:
        """ Element stored in the dictionary """
        def __init__(self, index, occurrences=1):
            """ Initialize Unit
            Args:
                index : index in the dictionary
                occurrences: actual number of counted occurrences (by default 1)
            """
            self._occurrences = occurrences
            self._index = index

        def update_occurrences(self, occurrences=1):
            self._occurrences += occurrences

        def change_index(self, new_index):
            self._index = new_index

        @property
        def occurrences(self):
            return self._occurrences

        def __eq__(self, other):
            return self._occurrences == other._occurrences and self._index == other._index

        def __ge__(self, other):
            """
            Guarantee that order of insertion is kept
            """
            return self._occurrences >= other._occurrences and self._index <= other._index

        def __str__(self):
            return f"ix:{self._index}\tocc:{self._occurrences}"

        @classmethod
        def from_str(cls, orig_str):
            tokens = orig_str.strip().split('\t')
            index = int(tokens[0].split(':')[1])
            occurrences = int(tokens[1].split(':')[1])
            return cls(index, occurrences)

    __UNK_TOKEN="N/A"
    __PAD_TOKEN="<<PAD>>"
    __BOS_TOKEN="<<BOS>>"
    __EOS_TOKEN="<<EOS>>"

    def __init__(self, initialize_special_tokens : bool = True):
        """ Initialize OccurrenceDict
        Args:
            initialize_special_tokens: if True then PAD, UNK, BOS and EOS tokens are
                                       inserted at the beginning of the dictionary
        """
        self._dict = {
            self.__PAD_TOKEN : self.Unit(0),
            self.__UNK_TOKEN : self.Unit(1),
            self.__BOS_TOKEN : self.Unit(2),
            self.__EOS_TOKEN : self.Unit(3)
        } if initialize_special_tokens else {}
        self.special_tokens = [ key for key in self._dict.keys() ]
        self._initialize_special_tokens = initialize_special_tokens

    def __getitem__(self, item):
        """ If item is not in the OccurrenceDict, special UNK token is returned"""
        if item in self._dict:
            return self._dict[item]
        else:
            return self._dict[self.__UNK_TOKEN]

    def __iter__(self):
        return self._dict.__iter__()

    def __contains__(self, item):
        return self._dict.__contains__(item)

    def get_pad(self):
        if self._initialize_special_tokens:
            return self.__PAD_TOKEN
        else:
            raise RuntimeError("Cannot return PAD_TOKEN when dict was initialized without special tokens")

    def get_eos(self):
        if self._initialize_special_tokens:
            return self.__EOS_TOKEN
        else:
            raise RuntimeError("Cannot return EOS_TOKEN when dict was initialized without special tokens")

    def get_bos(self):
        if self._initialize_special_tokens:
            return self.__BOS_TOKEN
        else:
            raise RuntimeError("Cannot return BOS_TOKEN when dict was initialized without special tokens")

    def get_unk(self):
        if self._initialize_special_tokens:
            return self.__UNK_TOKEN
        else:
            raise RuntimeError("Cannot return BOS_TOKEN when dict was initialized without special tokens")

    def pop(self, item):
        """
        safe pop, no penalty if item isn't in the dictionary
        :param item: key to be popped
        """
        if item in self._dict:
            return self._dict.pop(item)
        return None

    def add(self, word, occurrences=1):
        """ Adds the word to the OccurreceDict or increments its count if it is already in"""
        token = str(word)
        if token in self._dict:
            self._dict[word].update_occurrences(occurrences)
        else:
            self._dict[word] = OccurrenceDict.Unit(len(self._dict), occurrences)

    def sort(self, prunning : int = None, prun_occurrences : int = None):
        """
        Sorts the occurrence dict and returns it, NOT IN PLACE

        Args:
            prunning:         only leave prunning elements in the resulting dict
            prun_occurrences: all elements with less than prun_occurrences
                              occurrences will be removed from the resulting dict
        Returns:
            sorted dict
        """

        # do not sort special tokens, leave them out, they need to be on first positions
        for token in self.special_tokens:
            self._dict.pop(token)
        sorted_list = sorted(self._dict.items(), key=lambda item: item[1], reverse=True)
        for ix, token in enumerate(self.special_tokens):
            self._dict[token] = self.Unit(ix)

        if prunning is not None:
            sorted_list = sorted_list[:prunning]
        if prun_occurrences is not None:
            sorted_list = [ s for s in sorted_list if s[1].occurrences >= prun_occurrences]

        # special tokens are implicitly created in __init__
        result = OccurrenceDict()
        for _, (key, unit) in enumerate(sorted_list):
            result.add(key, unit.occurrences)  # no need to update index, the indices are automatically incremented
        return result

    def to_dict(self):
        result = {}
        for ix, (key, _) in enumerate(self._dict.items()):
            result[key] = ix
        return result

    def keys(self):
        return self._dict.keys()

    def save(self, file_path):
        """ Save the textual representation of the dict to a file"""
        with open(file_path, 'w') as f:
            for key in self.keys():
                print(f"{key}\t{self[key]}", file=f)

    def update(self, other, basic_dict : bool = False):
        """Update the OccurrenceDict with other dictionary

        Copies the keys along with their occurrence counts
        """
        for key in other.keys():
            if basic_dict:
                self.add(key, other[key])
            else:
                self.add(key, other[key].occurrences)
        return self

    @classmethod
    def load(cls, file_path, basic_dict : bool = False):
        """ Load the OccurrenceDict from the file_path
        Args:
            file_path  (str):  path to file with saved dictionary
            basic_dict (bool): whether the saved dict is the OccurrenceDict or not
        Returns:
            new OccurrenceDict loaded from the file_path"""
        with open(file_path, 'r', encoding='utf8') as f:
            file_content = f.read().strip().split('\n')
        result = cls()
        for _, line in enumerate(file_content):
            if basic_dict:
                # basic dict is saved in pairs "token occurrences"
                tokens = line.strip().split()
                word = tokens[0]
                occurrences = int(tokens[1])
                result.add(word, occurrences)
            else:
                # OccurrenceDict is saved in triples "token index occurrences"
                tokens = line.strip().split('\t')
                word = tokens[0]
                result._dict[word] = cls.Unit.from_str("\t".join(tokens[1:]))
        return result


def join_strings(first, second, *args, delimiter=" "):
    """
    join strings with delimiter, if any of strings is of
    zero length, then it is left ignored
    :return: joined string
    """
    result = ""
    if first != "":
        result = first
        if second != "":
            result = f"{first}{delimiter}{second}"
    elif second != "":
        result = second

    if len(args) > 0:
        return join_strings(result, *args)

    return result


class Logger:
    """ class that directly calls print() but allows to define if the call has the effect or not"""
    def __init__(self, log=True):
        self._log = log

    def switch_log(self):
        self._log = not self._log

    def set_on_log(self):
        self._log = True

    def set_off_log(self):
        self._log = False

    def __call__(self, message, file=sys.stderr):
        if self._log:
            print(message, file=file)

def set_home_away(home_city, away_city, actual_city):
    if actual_city == home_city:
        return "HOME"
    elif actual_city == away_city:
        return "AWAY"
    else:
        raise RuntimeError(f"NON VALID CITY NAME! {actual_city} : {home_city} : {away_city}")

city_transform_dict = { "Los Angeles" : "Los_Angeles", "LA" : "Los_Angeles" }

def transform_city_name(city_name):
    if city_name in city_transform_dict.keys():
        return city_transform_dict[city_name]
    return city_name

player_transform_dict = { "T.J. McConnell" : "TJ McConnell"}

def resolve_player_name_faults(player_name):
    """Applies name_transformations on all the tokens in the player names
    
    This is specifically aimed at cases like "J.J. Barea" where sometimes Barea
    is mentioned as "J.J." and sometimes as "JJ"
    """
    player_name_transformed = ""
    for token in player_name.strip().split():
        if token in name_transformations:
            player_name_transformed = join_strings(player_name_transformed, *name_transformations[token])
        else:
            player_name_transformed = join_strings(player_name_transformed, token)
    return player_name_transformed

def create_tp_vocab():
    """ Returns vocab of different types present in RotoWire tables"""
    type_dict = OccurrenceDict()

    for tp in BoxScoreEntries:
        type_dict.add(tp.value)
    for tp in LineScoreEntries:
        type_dict.add(tp.value)
    return type_dict

def create_ha_vocab():
    """ Returns vocab consisting of tokens PAD, UNK, EOS, BOS, HOME and AWAY"""
    _vocab = OccurrenceDict()
    _vocab.add("HOME")
    _vocab.add("AWAY")
    return _vocab