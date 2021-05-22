try:
    from constants import BoxScoreEntries, name_transformations # pylint: disable=import-error
    from utils import EnumDict, OccurrenceDict, set_home_away, transform_city_name, join_strings, \
        resolve_player_name_faults# pylint: disable=import-error
    from record_class import Record # pylint: disable=import-error
except:
    from .constants import BoxScoreEntries, name_transformations
    from .utils import EnumDict, OccurrenceDict, set_home_away, transform_city_name, join_strings, \
        resolve_player_name_faults
    from .record_class import Record

class BoxScore:
    def __init__( self
                , box_score_dict
                , home_city
                , away_city
                , player_dict
                , cell_dict
                , order_records=False
                , prun_records=False):
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
        pts_rec_pairs = []

        for player_number in self.get_player_numbers(self._dct):
            player_pts, player_records = self.extract_player_info( player_number
                                                                 , self._dct
                                                                 , home_city
                                                                 , away_city
                                                                 , player_dict
                                                                 , cell_dict)
            # the ordering is postponed until all the players are processed                             
            if order_records:
                pts_rec_pairs.append((player_pts, player_records))
            else:
                self._records += player_records
        
        # we order the players by their point totals during the matches
        if order_records:
            ix = 0
            for _, rec in sorted(pts_rec_pairs, key=lambda x: x[0], reverse=True):
                if prun_records:
                    if ix < 10:
                        self._records += self.filter_records(rec, ix < 3)
                    ix += 1
                else:
                    self._records += rec

    @staticmethod
    def filter_records( records
                      , advanced_stats : bool):
        to_be_kept = { BoxScoreEntries.ast
                     , BoxScoreEntries.min
                     , BoxScoreEntries.pts
                     , BoxScoreEntries.team_city
                     , BoxScoreEntries.player_name}
        to_be_prunned = { BoxScoreEntries.start_position
                        , BoxScoreEntries.second_name
                        , BoxScoreEntries.first_name}
        filtered_records = []
        if not advanced_stats:
            for r in records:
                if BoxScoreEntries(r.type) in to_be_kept:
                    filtered_records.append(r)
        else:
            for r in records:
                if BoxScoreEntries(r.type) not in to_be_prunned:
                    filtered_records.append(r)
        
        return filtered_records
        

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
        player_name = resolve_player_name_faults(player_name)
        entity_dict.add(player_name)
        if home_city == away_city:
            print(f"Couldn't resolve home_away ! ({home_city})")
        home_away = set_home_away(home_city, away_city, dct[BoxScoreEntries.team_city][player_number])

        # transform Los Angeles to Los_Angeles -> done after setting home_away, because
        # in the dataset LA and Los Angeles are used as different names for
        # Lakers and Clippers
        dct.mapmap(BoxScoreEntries.team_city, player_number, transform_city_name)
        dct[BoxScoreEntries.player_name][player_number] = player_name
        points_scored_str = dct[BoxScoreEntries.pts][player_number]
        points_scored = 0 if points_scored_str == "N/A" else int(points_scored_str)

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
        return (points_scored, records)

    @property
    def records(self):
        return self._records

    def get_player_list(self):
        return [value for _, value in self._dct[BoxScoreEntries.player_name].items()]
