from enum import Enum
from collections import OrderedDict
from information_extraction.constants import MatchStatEntries, LineScoreEntries, BoxScoreEntries, number_words
from information_extraction.utils import EnumDict, join_strings, OccurrenceDict
from text_to_num import text2num

import nltk.tokenize as nltk_tok
import numpy as np
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
name_transformations = {
    "T.J.": "TJ",
    "J.J.": "JJ",
    "C.J .": "CJ",
    "Steph": "Stephen"
}


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
                , player_dict
                , word_dict):
        """
        Creates the records from the BoxScore
        BoxScore contains information about all the players, their stats, which team they're part of
        - the information about one player is grouped in succeeding records
        """
        self._dct = EnumDict(box_score_dict)
        self._records = []

        for player_number in self.get_player_numbers(self._dct):
            self._records += self.extract_player_info( player_number
                                                     , self._dct
                                                     , home_city
                                                     , away_city
                                                     , player_dict
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
        player_name_transformed = ""
        for token in player_name.strip().split():
            if token in name_transformations:
                player_name_transformed = join_strings(player_name_transformed, name_transformations[token])
            else:
                player_name_transformed = join_strings(player_name_transformed, token)
        player_name = player_name_transformed
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

    def get_player_list(self):
        return [value for _, value in self._dct[BoxScoreEntries.player_name].items()]


class LineScore:
    def __init__( self
                , line_score_dict
                , home_city
                , away_city
                , city_dict
                , team_name_dict
                , word_dict):
        dct = EnumDict(line_score_dict)
        self._records = self.create_records(dct, home_city, away_city, city_dict, team_name_dict, word_dict)

    @staticmethod
    def create_records( dct
                      , home_city
                      , away_city
                      , city_dict
                      , team_name_dict
                      , word_dict):
        home_away = set_home_away(home_city, away_city, dct[LineScoreEntries.city])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.map(LineScoreEntries.city, transform_name)
        entity_name = dct[LineScoreEntries.name]
        team_name_dict.add(entity_name)
        city_dict.add(dct[LineScoreEntries.city])
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
    def traverse_span(span, entities : OccurrenceDict):
        """
        traverse span of word tokens until we find a word which isn't any entity
        :return: entity found in the span
        """
        candidate = span[0]
        index = 1
        while index < len(span) and join_strings(candidate, span[index]) in entities:
            candidate = join_strings(candidate, span[index])
            index += 1
        return index, candidate

    @staticmethod
    def extract_players(sentence, player_entities):
        """
        Traverse the sentence and try to extract all the
        named entities present in it
        :return: list with all the extracted named entities
        """
        index = 0
        tokenized_sentence = sentence.split()
        candidates = []
        while index < len(tokenized_sentence):
            if tokenized_sentence[index] in player_entities:
                i, candidate = Summary.traverse_span(tokenized_sentence[index:], player_entities)
                index += i
                candidates.append(candidate)
            else:
                index += 1

        return candidates

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

    def collect_tokens(self, word_dict : OccurrenceDict):
        for token in self._list_of_words:
            word_dict.add(token)

    @staticmethod
    def _transform_words(list_of_words):
        summary = join_strings(*list_of_words)
        sentences = [Summary.transform_numbers(s) for s in nltk_tok.sent_tokenize(summary)]
        result = []
        for s in sentences:
            tokens = s.strip().split()
            ix = 0
            while ix < len(tokens):
                two_tokens = " ".join(tokens[ix:ix+2])
                if tokens[ix] in name_transformations:
                    result.append(name_transformations[tokens[ix]])
                    ix += 1
                elif two_tokens in name_transformations:
                    result.append(name_transformations[two_tokens])
                    ix += 2
                else:
                    result.append(tokens[ix])
                    ix += 1

        return result

    def get_entities_from_summary(self, player_entities):
        summary = join_strings(*self._list_of_words)
        extracted = []
        for s in nltk_tok.sent_tokenize(summary):
            extracted += self.extract_players(s, player_entities)
        return extracted

    def transform(self, transformations):
        new_list_of_words = []
        ix = 0
        length = len(self._list_of_words)
        while ix < length:
            found = False
            for r in range(3, 0, -1):
                candidate = " ".join(self._list_of_words[ix:ix+r])
                if candidate in transformations:
                    ix += r
                    new_list_of_words.append(transformations[candidate])
                    found = True
                    break
            if not found:
                new_list_of_words.append(self._list_of_words[ix])
                ix += 1
        self._list_of_words = new_list_of_words

    def get_words(self):
        return self._list_of_words

    def __init__( self
                , list_of_words
                , word_dict):
        self._list_of_words = self._transform_words(list_of_words)
        self.collect_tokens(word_dict)

    def __str__(self):
        return " ".join(self._list_of_words)

    def __len__(self):
        return self._list_of_words.__len__()


class Logger:
    def __init__(self, log=True):
        self._log = log

    def __call__(self, message):
        if self._log:
            print(message)


class MatchStat:
    def __init__( self
                , match_dict
                , word_dict : OccurrenceDict
                , player_dict
                , city_dict
                , team_name_dict
                , cell_dict):
        dct = EnumDict(match_dict)
        if not self._is_summary_valid(dct):
            return
        home_city, vis_city = [ dct[key] for key in [MatchStatEntries.home_city, MatchStatEntries.vis_city]]
        self.box_score = BoxScore(
            dct[MatchStatEntries.box_score], home_city, vis_city, player_dict, cell_dict
        )
        self.home_line = LineScore(
            dct[MatchStatEntries.home_line], home_city, vis_city, city_dict, team_name_dict, cell_dict
        )
        self.vis_line = LineScore(
            dct[MatchStatEntries.vis_line], home_city, vis_city, city_dict, team_name_dict, cell_dict
        )
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

    for tp in BoxScoreEntries:
        type_dict.add(tp.value)
    for tp in LineScoreEntries:
        type_dict.add(tp.value)
    return type_dict


def extract_players_from_summaries(matches, player_dict, logger, transform_player_names=False):
    def dict_to_set(dct):
        result = set(dct.keys())
        for k in list(result):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        result.add(piece)
        return result

    player_in_summary_dict = OccurrenceDict()
    # create a player_set consisting of all important parts of player name
    # e.g. Tony Parker -> Tony, Parker, Tony Parker
    # important for extraction of player entities
    player_set = dict_to_set(player_dict)

    # collect all the players mentioned in unique summaries
    # collect all the parts of their name
    for match in matches:
        candidates_from_summary = set()
        unique_candidates_from_summary = set()
        # get all the entities from the summary
        entities = match.summary.get_entities_from_summary(player_set)
        box_players = set(match.box_score.get_player_list())
        for candidate in entities:
            candidates_from_summary.add(candidate)

        # merge anaphors "LeBron James" and "James" are the same if they're mentioned
        # in the same summary
        for candidate in list(candidates_from_summary):
            tokens = candidate.split()
            if len(tokens) == 1:
                found = False

                # try to substitue the token with entity name from the summary
                candidates_from_summary.remove(candidate)
                for c in candidates_from_summary:
                    if tokens[0] in c:
                        logger(f"Substituing {candidate} with {c} (in)")
                        found = True
                candidates_from_summary.add(candidate)

                # try to substitue the token with entity name from table statistics
                if not found:
                    for c in box_players:
                        if candidate in c:
                            unique_candidates_from_summary.add(c)
                            found = True
                            logger(f"Substituing {candidate} with {c} (out)")
                            break
                if not found:
                    logger(f"{candidate} unresolved")
            else:
                unique_candidates_from_summary.add(candidate)

        # add to dictionary all the occurrences of unique tokens
        transformations = {}
        for candidate in entities:
            if candidate in unique_candidates_from_summary:
                player_in_summary_dict.add(candidate)
                if candidate not in transformations:
                    transformations[candidate] = "_".join(candidate.strip().split())
                    logger(f"{candidate} -> {transformations[candidate]}")
            else:
                for c in unique_candidates_from_summary:
                    if candidate in c:
                        player_in_summary_dict.add(c)
                        if candidate not in transformations:
                            transformations[candidate] = "_".join(c.strip().split())
                            logger(f"{candidate} -> {transformations[candidate]}")
                        break

        if transform_player_names:
            match.summary.transform(transformations)

    return player_in_summary_dict


def extract_summaries_from_json(json_file_path, output_path, logger, transform_player_names=True):
    matches = []
    word_dict = OccurrenceDict()
    player_dict = OccurrenceDict()
    team_name_dict = OccurrenceDict()
    city_dict = OccurrenceDict()
    cell_dict = OccurrenceDict()

    with open(json_file_path, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match, word_dict, player_dict, city_dict, team_name_dict, cell_dict))
            if matches[-1].invalid:
                matches.pop()
                continue

    if transform_player_names:
        extract_players_from_summaries(matches, player_dict, logger, transform_player_names=transform_player_names)

    with open(output_path, 'w') as f:
        for match in matches:
            print(" ".join(match.summary.get_words()), file=f)


def gather_json_stats(json_file_path):
    """
    - traverse all the elements of the json,
    - extract all the match statistics and summaries
    - create dictionaries
    """
    matches = []
    word_dict = OccurrenceDict()
    player_dict = OccurrenceDict()
    team_name_dict = OccurrenceDict()
    city_dict = OccurrenceDict()
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
            matches.append(MatchStat(match, word_dict, player_dict, city_dict, team_name_dict, cell_dict))
            if matches[-1].invalid:
                matches.pop()
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

    player_in_summary_dict = extract_players_from_summaries(matches, player_dict)

    # print summary statistics
    print("---")
    print(f"total number of summaries : {len(matches)}")
    print(f"number of different tokens in summaries: {len(word_dict.keys())}")
    print(f"number of different tokens with more than 5 occurrences in summaries: {len(word_dict.sort(prun_occurrences=5).keys())}")
    print(f"max summary length : {max_summary_length}")
    print(f"min summary length : {min_summary_length}")
    print(f"average summary length : {total_summary_length / len(matches)}")

    # print record statistics
    print("---")
    print(f"max number of records : {max_table_length}")
    print(f"min number of records : {min_table_length}")
    print(f"average records length : {total_table_length / len(matches)}")

    # print other vocab statistics
    print("---")
    print(f"number of unique player names in summaries: {len(player_in_summary_dict.keys())}")
    print(f"number of unique player names in match stats: {len(player_dict.keys())}")
    print("---")
    more_than_five_1 = player_dict.sort(prun_occurrences=5)
    more_than_five_2 = player_in_summary_dict.sort(prun_occurrences=5)
    print(f"number of unique player names with more than or equal to 5 occurrences in summaries: {len(more_than_five_2.keys())}")
    print(f"number of unique player names with more than or equal to 5 occurrences in tables: {len(more_than_five_1.keys())}")
    print("---")
    print(f"number of different tokens in cell values : {len(cell_dict.keys())}")
    more_than_five_3 = cell_dict.sort(prun_occurrences=5)
    print(f"number of different tokens in cell values with more than or equal to 5 occurrences in cell values : {len(more_than_five_3.keys())}")
    print("---")
    print(f"number of unique city names in match stats : {len(city_dict.keys())}")
    print(f"number of unique team names in match stats : {len(team_name_dict.keys())}")
    print(f"number of different types of table cells : {len(type_dict.keys())}")

    # player statistics
    print("---")
    print("20 most mentioned players in the summaries")
    for player in player_in_summary_dict.sort(20).keys():
        print(player)


def _main():
    paths = ["rotowire/train.json", "rotowire/valid.json", "rotowire/test.json"]
    output_paths = [ "bpe_tests/train.txt", "bpe_tests/valid.txt", "bpe_tests/test.txt"]
    logger = Logger(log=False)
    for path, output_path in zip(paths, output_paths):
        print(f"path: {path}")
        extract_summaries_from_json(path, output_path, logger, transform_player_names=True)
        # gather_json_stats(path)


if __name__ == "__main__":
    _main()
