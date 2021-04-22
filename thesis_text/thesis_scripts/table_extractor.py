import csv
import json
from argparse import ArgumentParser

# class BoxScoreEntries(Enum):
#     ast = "AST"
#     blk = "BLK"
#     dreb = "DREB"
#     fg3a = "FG3A"
#     fg3m = "FG3M"
#     fg3_pct = "FG3_PCT"
#     fga = "FGA"
#     fgm = "FGM"
#     fg_pct = "FG_PCT"
#     first_name = "FIRST_NAME"
#     fta = "FTA"
#     ftm = "FTM"
#     ft_pct = "FT_PCT"
#     min = "MIN"
#     oreb = "OREB"
#     pf = "PF"
#     player_name = "PLAYER_NAME"
#     pts = "PTS"
#     reb = "REB"
#     second_name = "SECOND_NAME"
#     start_position = "START_POSITION"
#     stl = "STL"
#     team_city = "TEAM_CITY"
#     to = "TO"

def remove_min_dict(dct : dict):
    min = 500 # arbitrary high number
    to_remove = None
    for key in dct:
        if dct[key] < min:
            to_remove = key
            min = dct[key]
    dct.pop(to_remove)
    return dct

def _get_player_stats(matches):
    """
    Returns the keys of 4 best players in the game point-wise
    """
    match = matches[0]["box_score"]
    pts = dict([(key, int(value) if value != "N/A" else -1) for key, value in match["PTS"].items()])
    
    return [ str(k) for k in dict(sorted(pts.items(), key=lambda item: item[1], reverse=True)).keys() ]
                

def _create_player_table(args, matches):
    with open(args.output_path, 'w', encoding='utf8') as f:
        fieldnames = ['Name', 'Team City', 'S_POS', 'PTS', 'AST', 'REB', 'FG', 'FGA']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        best = _get_player_stats(matches)
        writer.writeheader()
        for val in best:
            writer.writerow({ 'Name' : matches[0]["box_score"]["PLAYER_NAME"][val]
                            , 'Team City' : matches[0]["box_score"]["TEAM_CITY"][val]
                            , 'S_POS' : matches[0]["box_score"]["START_POSITION"][val]
                            , 'PTS' : matches[0]["box_score"]["PTS"][val]
                            , 'AST' : matches[0]["box_score"]["AST"][val]
                            , 'REB' : matches[0]["box_score"]["REB"][val]
                            , 'FG' : matches[0]["box_score"]["FGM"][val]
                            , 'FGA' : matches[0]["box_score"]["FGA"][val]
            })


def _create_team_table(args, matches):
    with open(args.output_path, 'w', encoding='utf8') as f:
        fieldnames = ['Name', 'City', 'PTS', 'AST', 'REB', 'FG_PCT', 'Wins', 'Losses']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in ["home_line", "vis_line"]:
            writer.writerow({ 'Name'   : matches[0][entry]["TEAM-NAME"]
                            , 'City'   : matches[0][entry]["TEAM-CITY"]
                            , 'PTS'    : matches[0][entry]["TEAM-PTS"]
                            , 'AST'    : matches[0][entry]["TEAM-AST"]
                            , 'REB'    : matches[0][entry]["TEAM-REB"]
                            , 'FG_PCT'    : matches[0][entry]["TEAM-FG_PCT"]
                            , 'Wins'   : matches[0][entry]["TEAM-WINS"]
                            , 'Losses' : matches[0][entry]["TEAM-LOSSES"]
            })

def _create_summary(args, matches):
    with open(args.output_path, 'w', encoding='utf8') as f:
        match = matches[0]["summary"]
        f.write(" ".join(match))

def _main(args):
    with open(args.file_path, 'r', encoding='utf8') as f:
        matches = json.load(f, object_pairs_hook=dict)
    
    if args.team_stats:
        _create_team_table(args, matches)
    elif args.extract_summary:
        _create_summary(args, matches)
    else:
        _create_player_table(args, matches)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--team_stats", action='store_true')
    parser.add_argument("--extract_summary", action='store_true')
    _main(parser.parse_args())
