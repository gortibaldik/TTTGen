import json
from collections import OrderedDict
from match_stat_class import MatchStat # pylint: disable=import-error

def extract_matches_from_json(json_file_path, **kwargs):
    matches = []
    with open(json_file_path, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match, **kwargs))
            if matches[-1].invalid:
                matches.pop()
                continue
    return matches