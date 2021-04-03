from enum import Enum
from collections import OrderedDict
from information_extraction.constants import MatchStatEntries, LineScoreEntries, BoxScoreEntries, number_words
from information_extraction.utils import EnumDict, join_strings, OccurrenceDict, Logger
from text_to_num import text2num

import nltk.tokenize as nltk_tok
import numpy as np
import json
import argparse
import os


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
    "T.J.": ["TJ"],
    "J.J.": ["JJ"],
    "C.J .": ["CJ"],
    "J.R.": ["JR"],
    "K.J .": ["KJ"],
    "Steph Curry": ["Stephen", "Curry"],
    "Tony Douglas": ["Toney", "Douglas"],
    "Corey Joseph": ["Cory", "Joseph"],
    "Jose Barea": ["JJ", "Barea"],
    "Terrance Jones": ["Terrence", "Jones"],
    "Aaron Afflalo": ["Arron", "Afflalo"],
    "Andrew Brown": ["Anthony", "Brown"],
    "Dwyane Wade": ["Dwyane", "Wade"],
    "Jonathan Simmons": ["Jonathon", "Simmons"],
    "Mo Speights": ["Speights"],
    "Reggie Jefferson": ["Richard", "Jefferson"],
    "Luc Richard Mbah A Moute": ["Moute"],
    "Luc Mbah a Moute": ["Moute"],
    "Luc Mbah A Moute": ["Moute"],
    "Luc Richard Mbah a Moute": ["Moute"],
    "Mbah a Moute": ["Moute"],
    "Mbah A Moute": ["Moute"]
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
                , cell_dict):
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
                                                     , cell_dict)

    @staticmethod
    def get_player_numbers(dct : EnumDict):
        return dct[BoxScoreEntries.first_name].keys()

    @staticmethod
    def extract_player_info( player_number
                           , dct : EnumDict
                           , home_city : str
                           , away_city : str
                           , entity_dict : OccurrenceDict
                           , cell_dict : OccurrenceDict):
        records = []
        player_name = dct[BoxScoreEntries.player_name][player_number]
        player_name_transformed = ""
        for token in player_name.strip().split():
            if token in name_transformations:
                player_name_transformed = join_strings(player_name_transformed, *name_transformations[token])
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
            cell_dict.add(value)
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
                , cell_dict):
        dct = EnumDict(line_score_dict)
        self._records = self.create_records(dct, home_city, away_city, city_dict, team_name_dict, cell_dict)

    @staticmethod
    def create_records( dct
                      , home_city
                      , away_city
                      , city_dict
                      , team_name_dict
                      , cell_dict):
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
            cell_dict.add(value)
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
    def traverse_span(span, entities_set):
        """
        traverse span of word tokens until we find a word which isn't any entity
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
            tokens = []
            # transform possessives
            for token in s.strip().split():
                if token.endswith('’s'):
                    tokens.append(token.replace('’s', ''))
                    tokens.append("’s")
                else:
                    tokens.append(token)
            ix = 0
            # transform dataset faults
            while ix < len(tokens):
                found = False
                for r in range(5, 0, -1):
                    multi_tokens = " ".join(tokens[ix:ix+r])
                    if multi_tokens in name_transformations:
                        result += name_transformations[multi_tokens]
                        found = True
                        ix += r
                        break

                if not found:
                    result.append(tokens[ix])
                    ix += 1

        return result

    def get_entities_from_summary(self, entities_set):
        """
        Traverse the summary and try to extract all the named entities present in it
        - problem: all the substrings present in the summary must be in the entities_set, therefore
        if we search for "Luc Mbah a Moute" then {"Luc", "Luc Mbah", "Luc Mbah a", "Luc Mbah a Moute"} must
        be a subset of the entities set
        :return: list with all the extracted named entities
        """
        summary = join_strings(*self._list_of_words)
        extracted = []
        for s in nltk_tok.sent_tokenize(summary):
            extracted += self.extract_entities(s, entities_set)
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
                    if transformations[candidate] != "":
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


class MatchStat:
    _placeholder_dict = OccurrenceDict()

    def __init__( self
                , match_dict
                , player_dict : OccurrenceDict = _placeholder_dict
                , city_dict : OccurrenceDict = _placeholder_dict
                , team_name_dict : OccurrenceDict = _placeholder_dict
                , cell_dict : OccurrenceDict = _placeholder_dict
                , word_dict : OccurrenceDict = None
                , process_summary : bool = True):
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
        if process_summary:
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


def extract_matches_from_json(json_file_path, **kwargs):
    matches = []
    with open(json_file_path, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match, **kwargs))
            if matches[-1].invalid:
                matches.pop()
                continue
    return matches


def extract_players_from_summaries( matches
                                  , player_dict
                                  , logger
                                  , transform_player_names=False
                                  , prepare_for_bpe_training=False
                                  , prepare_for_bpe_application=False):
    def dict_to_set(dct):
        result = set(dct.keys())
        for k in list(result):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        result.add(piece)
        return result

    def log_if_important(logger, candidate):
        if candidate not in ["Orlando", "West", "Luke", "Christmas", "Bradley", "Harris",
                             "Michael", "Jordan", "Smart"]:
            logger(f"{candidate} unresolved")

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

                # try to substitute the token with entity name from the summary
                candidates_from_summary.remove(candidate)
                for c in candidates_from_summary:
                    if tokens[0] in c:
                        found = True
                candidates_from_summary.add(candidate)

                # try to substitute the token with entity name from table statistics
                if not found:
                    for c in box_players:
                        if candidate in c:
                            unique_candidates_from_summary.add(c)
                            found = True
                            break
                if not found:
                    log_if_important(logger, candidate)
            else:
                unique_candidates_from_summary.add(candidate)

        # add to dictionary all the occurrences of unique tokens
        transformations = {}
        for candidate in entities:
            if candidate in unique_candidates_from_summary:
                if candidate not in transformations:
                    transformations[candidate] = "_".join(candidate.strip().split())
                player_in_summary_dict.add(transformations[candidate])
            else:
                for c in unique_candidates_from_summary:
                    if candidate in c:
                        if candidate not in transformations:
                            transformations[candidate] = "_".join(c.strip().split())
                        player_in_summary_dict.add(transformations[candidate])
                        break

        if prepare_for_bpe_training:
            for key in transformations.keys():
                transformations[key] = ""

        if prepare_for_bpe_application:
            for key in transformations.keys():
                transformations[key] = f"<<<{transformations[key]}>>>"

        if transform_player_names or prepare_for_bpe_training or prepare_for_bpe_application:
            match.summary.transform(transformations)

    return player_in_summary_dict


def _prepare_for_create(args, set_names, input_paths):
    suffix = ""
    if args.to_npy and args.to_txt:
        raise RuntimeError("Multiple mutually exclusive options specified (--to_npy and --to_txt)")
    elif args.to_npy:
        suffix = ".npy"
    elif args.to_txt:
        suffix = ".txt"

    # prepare input paths
    for ix, pth in enumerate(set_names):
        input_paths[ix] = (os.path.join(args.preproc_summaries_dir, pth + "_prepared.txt"), input_paths[ix])

    # prepare output paths
    output_paths = []
    for ix, pth in enumerate(set_names):
        pth = os.path.join(args.output_dir, pth)
        output_paths.append((pth + "_in" + suffix, pth + "_target" + suffix))

    # prepare dictionary used on summaries
    entity_vocab_path = os.path.join(args.preproc_summaries_dir, "entity_vocab.txt")
    ent_vocab = OccurrenceDict.load(entity_vocab_path)
    token_vocab_path = os.path.join(args.preproc_summaries_dir, "token_vocab.txt")
    tk_vocab = OccurrenceDict.load(token_vocab_path, basic_dict=True)
    total_vocab = tk_vocab.update(ent_vocab).sort()

    # prepare dictionary on cell values
    cell_vocab_path = os.path.join(args.preproc_summaries_dir, "cell_vocab.txt")
    cl_vocab = OccurrenceDict.load(cell_vocab_path)

    # get max table length and max summary length
    mlt, mls = 0, 0
    with open(os.path.join(args.preproc_summaries_dir, "config.txt"), 'r') as f:
        tokens = [int(n) for n in f.read().strip().split('\n')]
        mlt, mls = tokens[0], tokens[1]
    return input_paths, output_paths, total_vocab, ent_vocab, cl_vocab, mlt, mls


def create_dataset( summary_path : str
                  , json_path : str
                  , out_in_path : str
                  , out_target_path : str
                  , summary_vocab
                  , cell_values_vocab
                  , named_entities_vocab
                  , max_summary_length
                  , max_table_length
                  , logger):

    def save_np_to_txt(_np_in: np.ndarray
                      , _np_target: np.ndarray
                      , _out_in_path: str
                      , _out_target_path: str
                      , _logger):
        _logger("--- saving to txt")
        with open(_out_in_path, 'w') as f:
            for _m_ix in range(_np_in.shape[0]):
                for _t_ix in range(_np_in.shape[2]):
                    for _v_ix in range(_np_in.shape[1]):
                        print(_np_in[_m_ix, _v_ix, _t_ix], end=" ", file=f)
                    print(end=";", file=f)
                print(file=f)

        with open(_out_target_path, 'w') as f:
            for _m_ix in range(_np_target.shape[0]):
                for _s_ix in range(_np_target.shape[1]):
                    print(_np_target[_m_ix, _s_ix], end=" ", file=f)
                print(file=f)

    tables = [ m.records for m in extract_matches_from_json( json_path
                                                           , word_dict=None
                                                           , process_summary=False)]
    logger(f"summaries {summary_path} -> {out_target_path}")
    logger(f"tables {json_path} -> {out_in_path}")

    with open(summary_path, 'r') as f:
        file_content = f.read().strip().split('\n')

    summaries = []
    for line in file_content:
        summaries.append(line)

    sum_to_ix = summary_vocab.to_dict()
    cell_to_ix = cell_values_vocab.to_dict()
    ent_to_ix = named_entities_vocab.to_dict()
    tp_to_ix = get_all_types().to_dict()
    ha_to_ix = { "HOME": 0, "AWAY" : 1}

    np_in = np.zeros(shape=[len(tables), 4, max_table_length])
    np_target = np.zeros(shape=[len(tables), max_summary_length])

    for m_ix, (table, summary) in enumerate(zip(tables, summaries)):
        for t_ix, record in enumerate(table):
            # zero reserved for padding
            np_in[m_ix, 0, t_ix] = tp_to_ix[record.type] + 1
            np_in[m_ix, 1, t_ix] = ent_to_ix["_".join(record.entity.strip().split())] + 1
            np_in[m_ix, 2, t_ix] = cell_to_ix[record.value] + 1
            np_in[m_ix, 3, t_ix] = ha_to_ix[record.ha] + 1

        for s_ix, subword in enumerate(summary.strip().split()):
            # zero reserved for padding
            np_target[m_ix, s_ix] = sum_to_ix[subword] + 1

    extension = os.path.splitext(out_in_path)[1]
    if extension == ".txt":
        save_np_to_txt(np_in, np_target, out_in_path, out_target_path, logger)
    elif extension == ".npy":
        np_in.save(out_in_path)
        np_target.save(out_target_path)
    elif extension == ".tfrecord":
        pass


def _prepare_for_extract(args, set_names):
    bpe_suffix = ""
    if args.prepare_for_bpe_training and args.prepare_for_bpe_application:
        print("Only one of --prepare_for_bpe_training --prepare_for_bpe_application can be used")
    elif args.prepare_for_bpe_training:
        bpe_suffix = "_pfbpe"  # prepared for bpe
    elif args.prepare_for_bpe_application:
        bpe_suffix = "_pfa"

    all_named_entities = None if args.entity_vocab_path is None else OccurrenceDict()
    cell_dict_overall = None if args.cell_vocab_path is None else OccurrenceDict()
    max_table_length = 0

    output_paths = []
    for name in set_names:
        output_paths.append(os.path.join(args.output_dir, name + bpe_suffix + args.file_suffix + ".txt"))
    return output_paths, all_named_entities, cell_dict_overall, max_table_length


def extract_summaries_from_json(json_file_path
                                , output_path
                                , logger
                                , transform_player_names=False
                                , prepare_for_bpe_training=False
                                , prepare_for_bpe_application=False
                                , all_named_entities: OccurrenceDict = None
                                , cell_dict_overall: OccurrenceDict = None):
    word_dict = OccurrenceDict()
    player_dict = OccurrenceDict()
    team_name_dict = OccurrenceDict()
    city_dict = OccurrenceDict()
    cell_dict = OccurrenceDict()

    matches = extract_matches_from_json( json_file_path
                                       , player_dict=player_dict
                                       , city_dict=city_dict
                                       , team_name_dict=team_name_dict
                                       , cell_dict=cell_dict
                                       , word_dict=word_dict)
    max_table_length = 0
    for match in matches:
       if max_table_length < len(match.records): max_table_length = len(match.records)

    if transform_player_names or prepare_for_bpe_training or prepare_for_bpe_application:
        tmp_dict = extract_players_from_summaries( matches
                                                 , player_dict
                                                 , logger
                                                 , transform_player_names=transform_player_names
                                                 , prepare_for_bpe_training=prepare_for_bpe_training
                                                 , prepare_for_bpe_application=prepare_for_bpe_application)

    # save named entities from the summaries and table
    if all_named_entities is not None:
        for key in tmp_dict.keys():
            all_named_entities.add(key, tmp_dict[key].occurrences)
        for key in team_name_dict.keys():
            # each city is mentioned 16 times in any vocab
            occurrences = team_name_dict[key].occurrences
            transformed = "_".join(key.strip().split())
            all_named_entities.add(transformed, occurrences)
        for key in player_dict.keys():
            transformed = "_".join(key.strip().split())
            if transformed not in all_named_entities:
                all_named_entities.add(transformed)

    # save cell values from the table
    if cell_dict_overall is not None:
        for key in cell_dict.keys():
            cell_dict_overall.add(key, cell_dict[key].occurrences)

    with open(output_path, 'w') as f:
        for match in matches:
            print(" ".join(match.summary.get_words()), file=f)

    return max_table_length


def gather_json_stats(json_file_path, logger, train_word_dict=None):
    """
    - traverse all the elements of the json,
    - extract all the match statistics and summaries
    - create dictionaries
    """
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

    matches = extract_matches_from_json( json_file_path
                                       , player_dict=player_dict
                                       , city_dict=city_dict
                                       , team_name_dict=team_name_dict
                                       , cell_dict=cell_dict
                                       , word_dict=word_dict)

    for match in matches:
        # collect summary statistics
        sum_length = len(match.summary)
        total_summary_length += sum_length
        if min_summary_length is None or sum_length < min_summary_length:
            min_summary_length = sum_length
        if max_summary_length is None or sum_length > max_summary_length:
            max_summary_length = sum_length

        # collect table statistics
        table_length = len(match.records)
        total_table_length += table_length
        if min_table_length is None or table_length < min_table_length:
            min_table_length = table_length
        if max_table_length is None or table_length > max_table_length:
            max_table_length = table_length

    ll = Logger(log=False)
    player_in_summary_dict = extract_players_from_summaries(matches, player_dict, ll)

    # print summary statistics
    logger("---")
    logger(f"total number of summaries : {len(matches)}")
    logger(f"max summary length : {max_summary_length}")
    logger(f"min summary length : {min_summary_length}")
    logger(f"average summary length : {total_summary_length / len(matches)}")
    logger("---")
    logger(f"number of different tokens in summaries: {len(word_dict.keys())}")
    logger(f"number of different tokens with more than 5 occurrences in summaries: {len(word_dict.sort(prun_occurrences=5).keys())}")
    if train_word_dict is None:
        train_word_dict = word_dict
    count = 0
    for word in word_dict.keys():
        if word in train_word_dict:
            count += 1
    overlap = (count * 100.0) / len(word_dict.keys())
    logger(f"percent of tokens from the train dict in the actual dict: {overlap}")

    # print record statistics
    logger("---")
    logger(f"max number of records : {max_table_length}")
    logger(f"min number of records : {min_table_length}")
    logger(f"average records length : {total_table_length / len(matches)}")

    # logger other vocab statistics
    logger("---")
    logger(f"number of unique player names in summaries: {len(player_in_summary_dict.keys())}")
    logger(f"number of unique player names in match stats: {len(player_dict.keys())}")
    logger("---")
    more_than_five_1 = player_dict.sort(prun_occurrences=5)
    more_than_five_2 = player_in_summary_dict.sort(prun_occurrences=5)
    logger(f"number of unique player names with more than or equal to 5 occurrences in summaries: {len(more_than_five_2.keys())}")
    logger(f"number of unique player names with more than or equal to 5 occurrences in tables: {len(more_than_five_1.keys())}")
    logger("---")
    logger(f"number of different tokens in cell values : {len(cell_dict.keys())}")
    more_than_five_3 = cell_dict.sort(prun_occurrences=5)
    logger(f"number of different tokens in cell values with more than or equal to 5 occurrences in cell values : {len(more_than_five_3.keys())}")
    logger("---")
    logger(f"number of unique city names in match stats : {len(city_dict.keys())}")
    logger(f"number of unique team names in match stats : {len(team_name_dict.keys())}")
    logger(f"number of different types of table cells : {len(type_dict.keys())}")

    # player statistics
    logger("---")
    logger("20 most mentioned players in the summaries")
    for player in player_in_summary_dict.sort(20).keys():
        logger(player)

    return word_dict


_extract_activity_descr = "extract_summaries"
_gather_stats_descr = "gather_stats"
_create_dataset_descr = "create_dataset"


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rotowire_dir",
        type=str,
        help="path to directory with original rotowire .json files \
              they should be found in the ${rotowire_dir}/{train, valid, test}.json",
        default="rotowire"
    )
    parser.add_argument(
        '--only_train',
        help='use only train data',
        action='store_true'
    )
    parser.add_argument(
        '--log',
        help='use extensive logging',
        action='store_true'
    )
    subparsers = parser.add_subparsers(
        title="activity",
        dest='activity',
        required=True,
        help='what to do ? <gather_stats|extract_summaries>'
    )
    gather_stats = subparsers.add_parser(_gather_stats_descr)
    gather_stats.add_argument(
        "--five_occurrences",
        help="After looking at training dataset, filter out all the tokens from the \
            dataset which occur less than 5 times",
        action='store_true'
    )
    extract_summaries_parser = subparsers.add_parser(_extract_activity_descr)
    extract_summaries_parser.add_argument(
        '--output_dir',
        type=str,
        help="directory where the outputs will be saved",
        required=True
    )
    extract_summaries_parser.add_argument(
        '--file_suffix',
        type=str,
        help="suffix appended after name of extracted summary\
         (train summary would be extracted to \"train_suffix.txt\")",
        default=""
    )
    extract_summaries_parser.add_argument(
        '--transform_players',
        help="transform names of players e.g. \"Stephen\" \"Curry\" to \"Stephen_Curry\"",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--prepare_for_bpe_training",
        help="extract all the player names from the text so that bpe isn't going to learn merging player names",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--prepare_for_bpe_application",
        help="prepare the input files for subword-nmt apply-bpe (change each player_token to <<<player_token>>> to be \
            able to use --glossaries \"<<<[^>]*>>>\" and don't change any of the player tokens)",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--entity_vocab_path",
        type=str,
        help="where to save the list of all the players mentioned in the summaries",
        default=None
    )
    extract_summaries_parser.add_argument(
        "--cell_vocab_path",
        type=str,
        help="where to save the list of all the cell values from the tables",
        default=None
    )
    extract_summaries_parser.add_argument(
        "--config_path",
        type=str,
        help="where to save the max_table_length",
        default=None
    )
    create_dataset_parser = subparsers.add_parser(_create_dataset_descr)
    create_dataset_parser.add_argument(
        "--preproc_summaries_dir",
        type=str,
        help="path to directory with output of create_dataset.sh",
        required=True
    )
    create_dataset_parser.add_argument(
        '--output_dir',
        type=str,
        help="directory where the outputs will be saved",
        required=True
    )
    create_dataset_parser.add_argument(
        "--to_npy",
        help="save the output to .npy format (indices in npy arrays)",
        action="store_true"
    )
    create_dataset_parser.add_argument(
        "--to_txt",
        help="save the outut to .txt format (indices in txt)",
        action="store_true"
    )
    return parser


def _main():
    parser = _create_parser()
    args = parser.parse_args()
    set_names = ["train", "valid", "test"]
    if args.only_train:
        set_names = [set_names[0]]
    input_paths = [ os.path.join(args.rotowire_dir, f + ".json") for f in set_names ]

    if args.activity == _extract_activity_descr:
        output_paths, all_named_entities, cell_dict_overall, max_table_length = _prepare_for_extract(args, set_names)
    elif args.activity == _create_dataset_descr:
        input_paths, output_paths, total_vocab, entity_vocab, cell_vocab, \
            max_table_length, max_summary_length= _prepare_for_create(args, set_names, input_paths)
    elif args.activity == _gather_stats_descr:
        output_paths = set_names

    logger = Logger(log=args.log)
    train_dict = None

    for input_path, output_path in zip(input_paths, output_paths):
        if args.activity == _extract_activity_descr:
            print(f"working with {input_path}, extracting to {output_path}")
            mtl = extract_summaries_from_json(
                input_path,
                output_path,
                logger,
                transform_player_names=args.transform_players,
                prepare_for_bpe_training=args.prepare_for_bpe_training,
                prepare_for_bpe_application=args.prepare_for_bpe_application,
                all_named_entities=all_named_entities,
                cell_dict_overall=cell_dict_overall
            )
            if mtl > max_table_length: max_table_length = mtl
        elif args.activity == _gather_stats_descr:
            print(f"working with {input_path}")
            if input_path == "rotowire/train.json":
                train_dict = gather_json_stats(input_path, logger)
                if args.five_occurrences:
                    train_dict = train_dict.sort(prun_occurrences=5)
            else:
                gather_json_stats(input_path, logger, train_dict)
        elif args.activity == _create_dataset_descr:
            summary_path = input_path[0]
            json_path = input_path[1]
            out_in_path = output_path[0]
            out_target_path = output_path[1]
            create_dataset(
                summary_path,
                json_path,
                out_in_path,
                out_target_path,
                total_vocab,
                cell_vocab,
                entity_vocab,
                max_summary_length=max_summary_length,
                max_table_length=max_table_length,
                logger=logger
            )

    if args.activity == _extract_activity_descr and args.entity_vocab_path is not None:
        all_named_entities.sort().save(args.entity_vocab_path)
    if args.activity == _extract_activity_descr and args.cell_vocab_path is not None:
        cell_dict_overall.sort().save(args.cell_vocab_path)
    if args.activity == _extract_activity_descr and args.config_path is not None:
        with open(args.config_path, "w") as f:
            print(max_table_length, file=f)


if __name__ == "__main__":
    _main()
