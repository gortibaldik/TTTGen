import json
from collections import OrderedDict
try:
    from match_stat_class import MatchStat # pylint: disable=import-error
except:
    from .match_stat_class import MatchStat

def extract_matches_from_json(json_file_path, **kwargs):
    """ Extract dict from json and traverse all the mentioned matches """
    matches = []
    with open(json_file_path, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match, **kwargs))
            if matches[-1].invalid:
                matches.pop()
                continue
    return matches