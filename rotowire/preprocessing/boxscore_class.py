from constants import BoxScoreEntries, name_transformations # pylint: disable=import-error
from utils import EnumDict, OccurrenceDict, set_home_away, transform_city_name, join_strings, transform_player_name # pylint: disable=import-error
from record_class import Record # pylint: disable=import-error

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
        # in our task neither first name, nor second name is needed or used
        self._dct.pop(BoxScoreEntries.first_name)
        self._dct.pop(BoxScoreEntries.second_name)
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
        return dct[BoxScoreEntries.player_name].keys()

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

        # transform Los Angeles to Los_Angeles -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.mapmap(BoxScoreEntries.team_city, player_number, transform_city_name)
        dct.mapmap(BoxScoreEntries.player_name, player_number, transform_player_name)

        for key in dct.keys():
            value = "_".join(dct[BoxScoreEntries(key)][player_number].strip().split())
            cell_dict.add(value)
            records.append(
                Record(
                    key,
                    player_name,
                    value,
                    home_away
                )
            )
        return records

    @property
    def records(self):
        return self._records

    def get_player_list(self):
        return [value for _, value in self._dct[BoxScoreEntries.player_name].items()]
