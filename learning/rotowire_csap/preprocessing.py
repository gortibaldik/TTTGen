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
                , entity_dict):
        """
        Creates the records from the BoxScore
        BoxScore contains information about all the players, their stats, which team they're part of
        - the information about one player is grouped in succeeding records
        """
        dct = EnumDict(box_score_dict)
        self._records = []

        for player_number in self.get_player_numbers(dct):
            self._records += self.extract_player_info(player_number, dct, home_city, away_city, entity_dict)

    @staticmethod
    def get_player_numbers(dct : EnumDict):
        return dct[BoxScoreEntries.first_name].keys()

    @staticmethod
    def extract_player_info( player_number
                           , dct : EnumDict
                           , home_city : str
                           , away_city : str
                           , entity_dict : OccurrenceDict):
        records = []
        player_name = dct[BoxScoreEntries.player_name][player_number]
        entity_dict.add(player_name)
        home_away = set_home_away(home_city, away_city, dct[BoxScoreEntries.team_city][player_number])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.mapmap(BoxScoreEntries.team_city, player_number, transform_name)

        for key in dct.keys():
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
                , entity_dict):
        dct = EnumDict(line_score_dict)
        self._records = self.create_records(dct, home_city, away_city, entity_dict)

    @staticmethod
    def create_records( dct
                      , home_city
                      , away_city
                      , entity_dict):
        home_away = set_home_away(home_city, away_city, dct[LineScoreEntries.city])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.map(LineScoreEntries.city, transform_name)
        entity_name = dct[LineScoreEntries.name]
        entity_dict.add(entity_name)
        records = []
        for key in dct.keys():
            records.append(
                Record(
                    key,
                    entity_name,
                    dct[LineScoreEntries(key)],
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


class MatchStat:
    def __init__( self
                , match_dict
                , word_dict : OccurrenceDict
                , entity_dict):
        dct = EnumDict(match_dict)
        if not self._is_summary_valid(dct):
            return
        home_city, vis_city = [ dct[key] for key in [MatchStatEntries.home_city, MatchStatEntries.vis_city]]
        self.box_score = BoxScore(dct[MatchStatEntries.box_score], home_city, vis_city, entity_dict)
        self.home_line = LineScore(dct[MatchStatEntries.home_line], home_city, vis_city, entity_dict)
        self.vis_line = LineScore(dct[MatchStatEntries.vis_line], home_city, vis_city, entity_dict)
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


def _main():
    paths = ["rotowire/train.json"]  # ,"rotowire/valid.json", "rotowire/test.json"]
    matches = []
    sum_box_length = 0
    sum_line_length = 0
    word_dict = OccurrenceDict()
    entity_dict = OccurrenceDict()

    for path in paths:
        with open(path, 'r', encoding='utf8') as f:
            for match in json.load(f, object_pairs_hook=OrderedDict):
                matches.append(MatchStat(match, word_dict, entity_dict))
                sum_box_length += len(matches[-1].box_score.records)
                sum_line_length += len(matches[-1].vis_line.records)
                sum_line_length += len(matches[-1].home_line.records)

    word_keys = word_dict.keys()
    entity_keys = entity_dict.keys()
    print(f"number of different tokens in summaries: {len(word_keys)}")
    word_dict.sort()
    print(f"number of different entities in table : {len(entity_keys)}")


if __name__ == "__main__":
    _main()
