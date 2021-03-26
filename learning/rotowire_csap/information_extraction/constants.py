from enum import Enum


number_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"}


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