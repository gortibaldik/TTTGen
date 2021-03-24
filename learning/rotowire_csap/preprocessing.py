from enum import Enum
from collections import OrderedDict
import json


class SpecialDict:
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


class MatchStatEntries(Enum):
    home_name = "home_name"
    box_score = "box_score"
    home_city = "home_city"
    vis_name = "vis_name"
    summary = "summary"
    vis_line = "vis_line"
    vis_city = "vis_city"
    day = "day"
    home_line = "home_line"


class LineScoreEntries(Enum):
    name = "TEAM-NAME"
    city = "TEAM-CITY"
    ast = "TEAM-AST"
    fg3 = "TEAM-FG3_PCT"
    fg = "TEAM-FG_PCT"
    ft = "TEAM-FT_PCT"
    loss = "TEAM-LOSSES"
    pts = "TEAM-PTS"
    qtr1 = "TEAM-PTS_QTR1"
    qtr2 = "TEAM-PTS_QTR2"
    qtr3 = "TEAM-PTS_QTR3"
    qtr4 = "TEAM-PTS_QTR4"
    reb = "TEAM-REB"
    tov = "TEAM-TOV"
    wins = "TEAM-WINS"


class BoxScoreEntries(Enum):
    ast = "AST"
    blk = "BLK"
    dreb = "DREB"
    fg3a = "FG3A"
    fg3m = "FG3M"
    fg3_pct = "FG3_PCT"
    fga = "FGA"
    fgm = "FGM"
    fg_pct = "FG_PCT"
    first_name = "FIRST_NAME"
    fta = "FTA"
    ftm = "FTM"
    ft_pct = "FT_PCT"
    min = "MIN"
    oreb = "OREB"
    pf = "PF"
    player_name = "PLAYER_NAME"
    pts = "PTS"
    reb = "REB"
    second_name = "SECOND_NAME"
    start_position = "START_POSITION"
    stl = "STL"
    team_city = "TEAM_CITY"
    to = "TO"


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
                , away_city):
        """
        Creates the records from the BoxScore
        BoxScore contains information about all the players, their stats, which team they're part of
        - the information about one player is grouped in succeeding records
        """
        dct = SpecialDict(box_score_dict)
        self._records = []

        for player_number in self.get_player_numbers(dct):
            self._records += self.extract_player_info(player_number, dct, home_city, away_city)

    @staticmethod
    def get_player_numbers(dct : SpecialDict):
        return dct[BoxScoreEntries.first_name].keys()

    @staticmethod
    def extract_player_info( player_number
                           , dct : SpecialDict
                           , home_city : str
                           , away_city : str):
        records = []
        player_name = dct[BoxScoreEntries.player_name][player_number]
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
    def __init__(self
                 , line_score_dict
                 , home_city
                 , away_city):
        dct = SpecialDict(line_score_dict)
        self._records = self.create_records(dct, home_city, away_city)

    @staticmethod
    def create_records(dct, home_city, away_city):
        home_away = set_home_away(home_city, away_city, dct[LineScoreEntries.city])

        # transform Los Angeles to LA -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.map(LineScoreEntries.city, transform_name)
        entity_name = dct[LineScoreEntries.name]
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
    def __init__( self
                , list_of_words):
        self._list_of_words = list_of_words


class MatchStat:
    def __init__( self
                , match_dict):
        dct = SpecialDict(match_dict)
        home_city, vis_city = [ dct[key] for key in [MatchStatEntries.home_city, MatchStatEntries.vis_city]]
        self.box_score = BoxScore(dct[MatchStatEntries.box_score], home_city, vis_city)
        self.home_line = LineScore(dct[MatchStatEntries.home_line], home_city, vis_city)
        self.vis_line = LineScore(dct[MatchStatEntries.vis_line], home_city, vis_city)
        self.home_name = dct[MatchStatEntries.home_name]
        self.vis_name = dct[MatchStatEntries.vis_name]



def _main():
    FILE_PATH = "train.json"
    matches = []
    sum_box_length = 0
    sum_line_length = 0
    with open(FILE_PATH, 'r', encoding='utf8') as f:
        for match in json.load(f, object_pairs_hook=OrderedDict):
            matches.append(MatchStat(match))
            sum_box_length += len(matches[-1].box_score.records)
            sum_line_length += len(matches[-1].vis_line.records)
            sum_line_length += len(matches[-1].home_line.records)
    print(matches)


if __name__ == "__main__":
    _main()
