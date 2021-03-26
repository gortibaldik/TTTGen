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

    def map(self, key, fn):
        self[key] = fn(self[key])

    def mapmap(self, key1, key2, fn):
        self[key1][key2] = fn(self[key1][key2])

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()


class OccurrenceDict:
    @total_ordering
    class Unit:
        def __init__(self, index, occurrences=1):
            self._occurrences = occurrences
            self._index = index

        def update_occurrences(self):
            self._occurrences += 1

        def change_index(self, new_index):
            self._index = new_index

        def __eq__(self, other):
            return self._occurrences == other._occurrences and self._index == other._index

        def __ge__(self, other):
            """
            Guarantee that order of insertion is kept
            """
            return self._occurrences >= other._occurrences and self._index <= other._index

        def __str__(self):
            return f"ix:{self._index};occ:{self._occurrences}"

    def __init__(self):
        self._dict = {}

    def add(self, word):
        token = str(word)
        if token in self._dict:
            self._dict[word].update_occurrences()
        else:
            self._dict[word] = OccurrenceDict.Unit(len(self._dict))

    def sort(self, prunning : int = None):
        sorted_list = sorted(self._dict.items(), key=lambda item: item[1], reverse=True)
        if prunning is not None:
            sorted_list = sorted_list[:prunning]
        result = {}
        for ix, (key, unit) in enumerate(sorted_list):
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