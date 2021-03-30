from collections import defaultdict
from functools import total_ordering


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
        self[key] = fn(self[key])

    def mapmap(self, key1, key2, fn):
        self[key1][key2] = fn(self[key1][key2])

    def keys(self):
        return self._dict.keys()


class OccurrenceDict:
    @total_ordering
    class Unit:
        def __init__(self, index, occurrences=1):
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
            return f"ix:{self._index};occ:{self._occurrences}"

    def __init__(self, special_tokens=None):
        self._dict = {}
        self._special_tokens = special_tokens

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return self._dict.__iter__()

    def __contains__(self, item):
        return self._dict.__contains__(item)

    def pop(self, item):
        """
        safe pop, no penalty if item isn't in the dictionary
        :param item: key to be popped
        """
        if item in self._dict:
            self._dict.pop(item)

    def add(self, word, occurrences=1):
        token = str(word)
        if token in self._dict:
            self._dict[word].update_occurrences(occurrences)
        else:
            self._dict[word] = OccurrenceDict.Unit(len(self._dict), occurrences)

    def sort(self, prunning : int = None, prun_occurrences : int = None):
        sorted_list = sorted(self._dict.items(), key=lambda item: item[1], reverse=True)
        if prunning is not None:
            sorted_list = sorted_list[:prunning]
        if prun_occurrences is not None:
            sorted_list = [ s for s in sorted_list if s[1].occurrences >= prun_occurrences]
        result = OccurrenceDict(self._special_tokens)
        for ix, (key, unit) in enumerate(sorted_list):
            result.add(key, unit.occurrences)  # no need to update index, the indices are automatically incremented
        return result

    def to_dict(self):
        result = {}
        for ix, (key, _) in enumerate(self._dict.items()):
            result[key] = ix
        return result

    def keys(self):
        return self._dict.keys()


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
