try:
    from constants import LineScoreEntries # pylint: disable=import-error
    from utils import EnumDict, OccurrenceDict, set_home_away, transform_city_name # pylint: disable=import-error
    from record_class import Record # pylint: disable=import-error
except:
    from .constants import LineScoreEntries
    from .utils import EnumDict, OccurrenceDict, set_home_away, transform_city_name
    from .record_class import Record

class LineScore:
    """ LineScore class holds records connected to a particular team"""
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
        """ Initialize LineScore

        dct:                the dictionary containing all the json objects connected to one team
        home_city:          name of the home city
        away_city:          name of the away city
        city_dict:          all the city names are appended to city_dict
        team_name_dict:     all team names are appended to team_name_dict
        cell_dict:          all the values in the cells from BoxScore, and both LineScores are appended
        """
        home_away = set_home_away(home_city, away_city, dct[LineScoreEntries.city])

        # transform Los Angeles to Los_Angeles -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.map(LineScoreEntries.city, transform_city_name)
        entity_name = dct[LineScoreEntries.name]
        team_name_dict.add(entity_name)
        city_dict.add(dct[LineScoreEntries.city])
        records = []
        for key in dct.keys():
            value = "_".join(dct[LineScoreEntries(key)].strip().split())
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
