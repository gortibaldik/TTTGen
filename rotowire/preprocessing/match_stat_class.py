try:
    from constants import MatchStatEntries # pylint: disable=import-error
    from utils import EnumDict, OccurrenceDict # pylint: disable=import-error
    from summary_class import Summary # pylint: disable=import-error
    from boxscore_class import BoxScore # pylint: disable=import-error
    from linescore_class import LineScore # pylint: disable=import-error
    from record_class import Record # pyling: disable=import-error
except:
    from .constants import MatchStatEntries
    from .utils import EnumDict, OccurrenceDict
    from .summary_class import Summary
    from .boxscore_class import BoxScore
    from .linescore_class import LineScore
    from .record_class import Record

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
                , words_limit : int = None
                , order_records : bool = False
                , prun_records : bool = False):
        dct = EnumDict(match_dict)
        if not self._is_summary_valid(dct):
            return
        home_city = dct[MatchStatEntries.home_city]
        vis_city = dct[MatchStatEntries.vis_city]
        self.box_score = BoxScore( dct[MatchStatEntries.box_score]
                                 , home_city
                                 , vis_city
                                 , player_dict
                                 , cell_dict
                                 , order_records=order_records
                                 , prun_records=prun_records)
        self.home_line = LineScore( dct[MatchStatEntries.home_line]
                                  , home_city
                                  , vis_city
                                  , city_dict
                                  , team_name_dict
                                  , cell_dict)
        self.vis_line = LineScore( dct[MatchStatEntries.vis_line]
                                 , home_city
                                 , vis_city
                                 , city_dict
                                 , team_name_dict
                                 , cell_dict)
        self.home_name = dct[MatchStatEntries.home_name]
        self.vis_name = dct[MatchStatEntries.vis_name]
        od = OccurrenceDict()
        bos_value = od.get_bos()
        eos_value = od.get_eos()
        bos_record = Record(bos_value, bos_value, bos_value, bos_value)
        eos_record = Record(eos_value, eos_value, eos_value, eos_value)
        if order_records:
            self.records = [bos_record, eos_record] + self.home_line.records +\
                self.vis_line.records + self.box_score.records    
        else:
            self.records = [bos_record, eos_record] + self.box_score.records +\
                self.home_line.records + self.vis_line.records
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