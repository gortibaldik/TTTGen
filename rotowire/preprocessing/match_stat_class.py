try:
    from constants import MatchStatEntries # pylint: disable=import-error
    from utils import EnumDict, OccurrenceDict # pylint: disable=import-error
    from summary_class import Summary # pylint: disable=import-error
    from boxscore_class import BoxScore # pylint: disable=import-error
    from linescore_class import LineScore # pylint: disable=import-error
except:
    from .constants import MatchStatEntries
    from .utils import EnumDict, OccurrenceDict
    from .summary_class import Summary
    from .boxscore_class import BoxScore
    from .linescore_class import LineScore

class MatchStat:
    _placeholder_dict = OccurrenceDict()

    def __init__( self
                , match_dict
                , player_dict : OccurrenceDict = _placeholder_dict
                , city_dict : OccurrenceDict = _placeholder_dict
                , team_name_dict : OccurrenceDict = _placeholder_dict
                , cell_dict : OccurrenceDict = _placeholder_dict
                , word_dict : OccurrenceDict = None
                , process_summary : bool = True
                , words_limit : int = None):
        dct = EnumDict(match_dict)
        if not self._is_summary_valid(dct):
            return
        home_city = dct[MatchStatEntries.home_city]
        vis_city = dct[MatchStatEntries.vis_city]
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
            self.summary = Summary( dct[MatchStatEntries.summary]
                                  , word_dict
                                  , words_limit=words_limit)

    def _is_summary_valid(self, dct):
        if "Lorem" in dct[MatchStatEntries.summary]:
            for attr in ["box_score", "home_line", "vis_line", "home_name", "vis_name", "records", "summary"]:
                super().__setattr__(attr, None)
            self.invalid = True
            return False
        self.invalid = False
        return True