from enum import Enum
from collections import OrderedDict
from information_extraction.constants import MatchStatEntries, LineScoreEntries, BoxScoreEntries, number_words
from information_extraction.utils import EnumDict, join_strings, OccurrenceDict
from text_to_num import text2num

import nltk.tokenize as nltk_tok
import json


class Record:
    def __init__( self
                , type
                , entity
                , value
                , ha):
        self.type = type
        self.entity = entity
        self.value = value
        self.ha = ha

    def __str__(self):
        return f"{self.value} | {self.entity} | {self.type} | {self.ha}"


city_transform_dict = { "Los Angeles" : "LA" }


def transform_name(city_name):
    if city_name in city_transform_dict.keys():
        return city_transform_dict[city_name]
    return city_name


def set_home_away(home_city, away_city, actual_city):
    if actual_city == home_city:
        return "HOME"
    elif actual_city == away_city:
        return "AWAY"
    else:
        raise RuntimeError(f"NON VALID CITY NAME! {actual_city} : {home_city} : {away_city}")


class BoxScore:
    def __init__( self
                , box_score_dict
                , home_city
                , away_city
                , entity_dict
                , word_dict):
        """
        Creates the records from the BoxScore
        BoxScore contains information about all the players, their stats, which team they're part of
        - the information about one player is grouped in succeeding records
        """
        dct = EnumDict(box_score_dict)
        self._records = []

        for player_number in self.get_player_numbers(dct):
            self._records += self.extract_player_info( player_number
                                                     , dct, home_city
                                                     , away_city
                                                     , entity_dict
                                                     , word_dict)

    @staticmethod
    def get_player_numbers(dct : EnumDict):
        return dct[BoxScoreEntries.first_name].keys()

    @staticmethod
    def extract_player_info( player_number
                           , dct : EnumDict
                           , home_city : str
                           , away_city : str
                           , entity_dict : OccurrenceDict
                           , word_dict : OccurrenceDict):
        records = []
        player_name = dct[BoxScoreEntries.player_name][player_number]
        entity_dict.add(player_name)
        home_away = set_home_away(home_city, away_city, dct[BoxScoreEntries.team_city][player_number])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.mapmap(BoxScoreEntries.team_city, player_number, transform_name)

        for key in dct.keys():
            value = dct[BoxScoreEntries(key)][player_number]
            word_dict.add(value)
            records.append(
                Record(
                    key,
                    player_name,
                    dct[BoxScoreEntries(key)][player_number],
                    home_away
                )
            )
        return records

    @property
    def records(self):
        return self._records


class LineScore:
    def __init__( self
                , line_score_dict
                , home_city
                , away_city
                , entity_dict
                , word_dict):
        dct = EnumDict(line_score_dict)
        self._records = self.create_records(dct, home_city, away_city, entity_dict, word_dict)

    @staticmethod
    def create_records( dct
                      , home_city
                      , away_city
                      , entity_dict
                      , word_dict):
        home_away = set_home_away(home_city, away_city, dct[LineScoreEntries.city])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.map(LineScoreEntries.city, transform_name)
        entity_name = dct[LineScoreEntries.name]
        entity_dict.add(entity_name)
        records = []
        for key in dct.keys():
            value = dct[LineScoreEntries(key)]
            word_dict.add(value)
            records.append(
                Record(
                    key,
                    entity_name,
                    value,
                    home_away
                )
            )
        return records

    @property
    def records(self):
        return self._records


class Summary:
    @staticmethod
    def transform_numbers(sent):
        def has_to_be_ignored(__sent, __i):
            ignores = {"three point", "three - point", "three - pt", "three pt", "three - pointers", "three - pointer",
                       "three pointers"}
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

    @staticmethod
    def collect_tokens(word_dict : OccurrenceDict, sentences):
        for s in sentences:
            tokens = s.split()
            for token in tokens:
                word_dict.add(token)

    def __init__( self
                , list_of_words
                , word_dict):
        self._list_of_words = list_of_words
        summary = join_strings(*list_of_words)
        sentences = [self.transform_numbers(s) for s in nltk_tok.sent_tokenize(summary)]
        self.collect_tokens(word_dict, sentences)
        self._list_of_words = [ word for s in sentences for word in s.split()]

    def __str__(self):
        return " ".join(self._list_of_words)

    def __len__(self):
        return self._list_of_words.__len__()


class MatchStat:
    def __init__( self
                , match_dict
                , word_dict : OccurrenceDict
                , entity_dict
                , cell_dict):
        dct = EnumDict(match_dict)
        if not self._is_summary_valid(dct):
            return
        home_city, vis_city = [ dct[key] for key in [MatchStatEntries.home_city, MatchStatEntries.vis_city]]
        self.box_score = BoxScore(dct[MatchStatEntries.box_score], home_city, vis_city, entity_dict, cell_dict)
        self.home_line = LineScore(dct[MatchStatEntries.home_line], home_city, vis_city, entity_dict, cell_dict)
        self.vis_line = LineScore(dct[MatchStatEntries.vis_line], home_city, vis_city, entity_dict, cell_dict)
        self.home_name = dct[MatchStatEntries.home_name]
        self.vis_name = dct[MatchStatEntries.vis_name]
        self.records = self.box_score.records + self.home_line.records + self.vis_line.records
        self.summary = Summary(dct[MatchStatEntries.summary], word_dict)

    def _is_summary_valid(self, dct):
        if "Lorem" in dct[MatchStatEntries.summary]:
            for attr in ["box_score", "home_line", "vis_line", "home_name", "vis_name", "records", "summary"]:
                super().__setattr__(attr, None)
            self.invalid = True
            return False
        self.invalid = False
        return True


def get_all_types():
    type_dict = OccurrenceDict()

    for type in BoxScoreEntries:
        type_dict.add(type.value)
    for type in LineScoreEntries:
        type_dict.add(type.value)
    return type_dict


def create_dataset_from_json(json_file_path):
    """
    - traverse all the elements of the json,
    - extract all the match statistics and summaries
    - create dictionaries
    """
    matches = []
    word_dict = OccurrenceDict()
    entity_dict = OccurrenceDict()
    cell_dict = OccurrenceDict()
    type_dict = get_all_types()

    total_summary_length = 0
    max_summary_length = None
    min_summary_length = None

    total_table_length = 0
    max_table_length = None
    min_table_length = None

    with open(json_file_path, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match, word_dict, entity_dict, cell_dict))
            if matches[-1].invalid:
                continue

            # collect summary statistics
            sum_length = len(matches[-1].summary)
            total_summary_length += sum_length
            if min_summary_length is None or sum_length < min_summary_length:
                min_summary_length = sum_length
            if max_summary_length is None or sum_length > max_summary_length:
                max_summary_length = sum_length

            # collect table statistics
            table_length = len(matches[-1].records)
            total_table_length += table_length
            if min_table_length is None or table_length < min_table_length:
                min_table_length = table_length
            if max_table_length is None or table_length > max_table_length:
                max_table_length = table_length

    # print summary statistics
    print(f"number of different tokens in summaries: {len(word_dict.keys())}")
    print(f"max summary length : {max_summary_length}")
    print(f"min summary length : {min_summary_length}")
    print(f"average summary length : {total_summary_length / len(matches)}")
    print("---")

    # print record statistics
    print(f"max number of records : {max_table_length}")
    print(f"min number of records : {min_table_length}")
    print(f"average records length : {total_table_length / len(matches)}")
    print("---")

    # print other vocab statistics
    print(f"number of different entities in table : {len(entity_dict.keys())}")
    print(f"number of different tokens in cell values : {len(cell_dict.keys())}")
    print(f"number of different types of table cells : {len(type_dict.keys())}")


def _main():
    paths = ["rotowire/train.json"]  # , "rotowire/valid.json", "rotowire/test.json"]
    for path in paths:
        create_dataset_from_json(path)


if __name__ == "__main__":
    _main()
