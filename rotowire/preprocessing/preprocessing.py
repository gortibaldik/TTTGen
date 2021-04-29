from enum import Enum
from text_to_num import text2num
from collections import OrderedDict
if __name__ == '__main__':
    from constants import city_names, team_names # pylint: disable=import-error
    from utils import OccurrenceDict, Logger, create_tp_vocab # pylint: disable=import-error
    from extract_matches import extract_matches_from_json # pylint: disable=import-error
    from create_dataset import create_prepare, create_dataset, \
        create_dataset_parser, _create_dataset_descr # pylint: disable=import-error
else:
    from .constants import city_names, team_names
    from .utils import OccurrenceDict, Logger, create_tp_vocab
    from .extract_matches import extract_matches_from_json
    from .create_dataset import create_prepare, create_dataset, \
        create_dataset_parser, _create_dataset_descr


import nltk.tokenize as nltk_tok
import numpy as np
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default
import tensorflow as tf
import json
import argparse



def extract_players_from_summaries( matches
                                  , player_dict
                                  , logger
                                  , transform_player_names=False
                                  , prepare_for_bpe_training=False
                                  , prepare_for_bpe_application=False
                                  , lowercase=False
                                  , exception_cities=False
                                  , exception_teams=False):
    def dict_to_set(dct):
        not_names = ["IV", "II", "III", "Jr.", "Jr"]
        result = set(dct.keys())
        for k in list(result):
            pieces = k.split()
            if len(pieces) > 1:
                if pieces[-1] in not_names:
                    result.add(" ".join(pieces[:-1]))
                for piece in pieces:
                    if len(piece) > 1 and piece not in not_names:
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

    def add_modifier(candidate):
        players_with_modifiers = [
            "Kelly Oubre Jr.",
            "Ronald Roberts Jr.",
            "Wade Baldwin IV",
            "Johnny O'Bryant III",
            "Otto Porter Jr.",
            "Larry Drew II",
            "Glenn Robinson III",
            "James Ennis III",
            "Derrick Jones, Jr.",
            "Tim Hardaway Jr.",
            "Larry Nance Jr.",
            "John Lucas III",
            "Perry Jones III"
        ]
        for p in players_with_modifiers:
            if candidate in p:
                logger(f"{candidate} -> {p}")
                return p
        return candidate

    # collect all the players mentioned in unique summaries
    # collect all the parts of their name
    for _, match in enumerate(matches):
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
                    if tokens[0] in  " ".join(c.split()[1:]):
                        found = True
                candidates_from_summary.add(candidate)

                # try to substitute the token with entity name from table statistics
                if not found:
                    for c in box_players:
                        if candidate in " ".join(c.split()[1:]):
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
                    transformations[candidate] = "_".join(add_modifier(candidate).strip().split())
                player_in_summary_dict.add(transformations[candidate])
            else:
                for c in unique_candidates_from_summary:
                    if candidate in " ".join(c.split()[1:]):
                        if candidate not in transformations:
                            transformations[candidate] = "_".join(add_modifier(c).strip().split())
                        player_in_summary_dict.add(transformations[candidate])
                        break
        #TODO : DRY
        if prepare_for_bpe_training:
            for key in transformations.keys():
                transformations[key] = ""

            if exception_cities:
                for city_name in city_names:
                    transformations[city_name] = f""

            if exception_teams:
                for team_name in team_names:
                    transformations[team_name] = f""

        if prepare_for_bpe_application:
            for key in transformations.keys():
                transformations[key] = f"<<<{transformations[key]}>>>"

            if exception_cities:
                for city_name in city_names:
                    transformations[city_name] = f"<<<{city_name}>>>"

            if exception_teams:
                for team_name in team_names:
                    transformations[team_name] = f"<<<{team_name}>>>"

        if transform_player_names or prepare_for_bpe_training or prepare_for_bpe_application:
            match.summary.transform(transformations, lowercase)

    return player_in_summary_dict


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


def extract_summaries_from_json( json_file_path
                               , output_path
                               , logger
                               , transform_player_names=False
                               , prepare_for_bpe_training=False
                               , prepare_for_bpe_application=False
                               , lowercase=False
                               , exception_cities=False
                               , exception_teams=False
                               , words_limit=None
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
                                       , word_dict=word_dict
                                       , words_limit=words_limit)
    max_table_length = 0
    for match in matches:
       if max_table_length < len(match.records): max_table_length = len(match.records)

    if transform_player_names or prepare_for_bpe_training or prepare_for_bpe_application or lowercase:
        tmp_dict = extract_players_from_summaries( matches
                                                 , player_dict
                                                 , logger
                                                 , transform_player_names=transform_player_names
                                                 , prepare_for_bpe_training=prepare_for_bpe_training
                                                 , prepare_for_bpe_application=prepare_for_bpe_application
                                                 , lowercase=lowercase
                                                 , exception_cities=exception_cities
                                                 , exception_teams=exception_teams)

    count = 0
    # save named entities from the summaries and table
    if all_named_entities is not None:
        for key in tmp_dict.keys():
            if key not in all_named_entities:
                count +=1
            all_named_entities.add(key, tmp_dict[key].occurrences)
        for key in team_name_dict.keys():
            if key not in all_named_entities:
                count += 1
            # each city is mentioned 16 times in any vocab
            occurrences = team_name_dict[key].occurrences
            transformed = "_".join(key.strip().split())
            all_named_entities.add(transformed, occurrences)
        for key in player_dict.keys():
            transformed = "_".join(key.strip().split())
            if transformed not in all_named_entities:
                count += 1
                all_named_entities.add(transformed)
    logger(f"{count} new values introduced to all_named_entities")

    count = 0
    # save cell values from the table
    if cell_dict_overall is not None:
        for key in cell_dict.keys():
            if key not in cell_dict_overall:
                count += 1
            cell_dict_overall.add(key, cell_dict[key].occurrences)
    logger(f"{count} new values introduced to cell_dict_overall")

    with open(output_path, 'w') as f:
        for match in matches:
            print(" ".join(match.summary.get_words()), file=f)

    return max_table_length


def gather_json_stats(json_file_path, logger, train_word_dict=None, transform_player_names : bool = False):
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
    type_dict = create_tp_vocab()

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

    ll = Logger(log=False)
    player_in_summary_dict = extract_players_from_summaries( matches
                                                           , player_dict
                                                           , ll
                                                           , transform_player_names=transform_player_names)
    
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
        "--only_set",
        type=str,
        help="specify the set to be processed",
        default=None
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
    gather_stats.add_argument(
        "--transform_players",
        help="Gather stats over transformed player names",
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
        "--lowercase",
        help="lowercase all the tokens in the summaries except for the special ones",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--exception_cities",
        help="apply bpe-application or bpe-training transformations also to city names",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--exception_teams",
        help="apply bpe-application or bpe-training transformations also to team names",
        action='store_true'
    )
    extract_summaries_parser.add_argument(
        "--words_limit",
        help="limits the size of the extracted summaries",
        default=None,
        type=int
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
    create_dataset_parser(subparsers)
    return parser


def _main():
    parser = _create_parser()
    args = parser.parse_args()
    set_names = ["train", "valid", "test"]
    if args.only_train:
        set_names = [set_names[0]]
    if args.only_set is not None:
        set_names = [args.only_set]
    input_paths = [ os.path.join(args.rotowire_dir, f + ".json") for f in set_names ]

    if args.activity == _extract_activity_descr:
        output_paths, all_named_entities, cell_dict_overall, max_table_length = _prepare_for_extract(args, set_names)
    elif args.activity == _create_dataset_descr:
        input_paths, output_paths, total_vocab, max_table_length, \
            max_summary_length, max_plan_length = create_prepare(args, set_names, input_paths)
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
                exception_cities=args.exception_cities,
                exception_teams=args.exception_teams,
                lowercase=args.lowercase,
                words_limit=args.words_limit,
                all_named_entities=all_named_entities,
                cell_dict_overall=cell_dict_overall
            )
            if mtl > max_table_length: max_table_length = mtl
        elif args.activity == _gather_stats_descr:
            print(f"working with {input_path}")
            if os.path.basename(input_path) == "train.json":
                train_dict = gather_json_stats(input_path, logger, transform_player_names=args.transform_players)
                if args.five_occurrences:
                    train_dict = train_dict.sort(prun_occurrences=5)
            else:
                gather_json_stats(input_path, logger, train_dict, transform_player_names=args.transform_players)
        elif args.activity == _create_dataset_descr:
            create_dataset(
                input_path,
                output_path,
                total_vocab,
                max_plan_length=max_plan_length,
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
