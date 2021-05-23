import os

from argparse import ArgumentParser
from preprocessing.load_dataset import load_tf_record_dataset, load_values_from_config
from preprocessing.constants import LineScoreEntries

figure_template = """%%% The main file. It contains definitions of basic parameters and includes all other parts.

%% Settings for single-side (simplex) printing
% Margins: left 40mm, right 25mm, top and bottom 25mm
% (but beware, LaTeX adds 1in implicitly)
\\documentclass[12pt,a4paper]{{report}}
\\setlength\\textwidth{{145mm}}
\\setlength\\textheight{{247mm}}
\\setlength\\oddsidemargin{{15mm}}
\\setlength\\evensidemargin{{15mm}}
\\setlength\\topmargin{{0mm}}
\\setlength\\headsep{{0mm}}
\\setlength\\headheight{{0mm}}
% \\openright makes the following text appear on a right-hand page
\\let\\openright=\\clearpage

%% Settings for two-sided (duplex) printing
% \\documentclass[12pt,a4paper,twoside,openright]{{report}}
% \\setlength\\textwidth{{145mm}}
% \\setlength\\textheight{{247mm}}
% \\setlength\\oddsidemargin{{14.2mm}}
% \\setlength\\evensidemargin{{0mm}}
% \\setlength\\topmargin{{0mm}}
% \\setlength\\headsep{{0mm}}
% \\setlength\\headheight{{0mm}}
% \\let\\openright=\\cleardoublepage

%% Generate PDF/A-2u
\\usepackage[a-2u]{{pdfx}}

%% Character encoding: usually latin2, cp1250 or utf8:
\\usepackage[utf8]{{inputenc}}

%% Prefer Latin Modern fonts
\\usepackage{{lmodern}}

%% Further useful packages (included in most LaTeX distributions)
\\usepackage{{amsmath}}        % extensions for typesetting of math
\\usepackage{{amsfonts}}       % math fonts
\\usepackage{{amsthm}}         % theorems, definitions, etc.
\\usepackage{{bbding}}         % various symbols (squares, asterisks, scissors, ...)
\\usepackage{{bm}}             % boldface symbols (\\bm)
\\usepackage{{graphicx}}       % embedding of pictures
\\usepackage{{fancyvrb}}       % improved verbatim environment
\\usepackage{{natbib}}         % citation style AUTHOR (YEAR), or AUTHOR [NUMBER]
\\usepackage[nottoc]{{tocbibind}} % makes sure that bibliography and the lists
			    % of figures/tables are included in the table
			    % of contents
\\usepackage{{dcolumn}}        % improved alignment of table columns
\\usepackage{{booktabs}}       % improved horizontal lines in tables
\\usepackage{{paralist}}       % improved enumerate and itemize
\\usepackage{{xcolor}}         % typesetting in color
\\usepackage{{soulutf8}}			% highlighting text

% -----
% just some workarounds for soul to cooperate with xcolor
% taken from https://tex.stackexchange.com/questions/48501/soul-broken-highlighting-with-xcolor-when-using-selectcolormodel
\\usepackage{{etoolbox}}
\\makeatletter
\\patchcmd{{\\SOUL@ulunderline}}{{\\dimen@}}{{\\SOUL@dimen}}{{}}{{}}
\\patchcmd{{\\SOUL@ulunderline}}{{\\dimen@}}{{\\SOUL@dimen}}{{}}{{}}
\\patchcmd{{\\SOUL@ulunderline}}{{\\dimen@}}{{\\SOUL@dimen}}{{}}{{}}
\\newdimen\\SOUL@dimen
\\makeatother
% -----

\\usepackage{{tikz}}
\\usepackage{{subcaption}}
\\usepackage[linesnumbered,vlined,ruled]{{algorithm2e}}    % pseudo - codes
\\usepackage{{listings}} % python code in latex
\\usetikzlibrary{{positioning}}

% declaring argmin
\\DeclareMathOperator*{{\\argmax}}{{arg\\,max}}
\\DeclareMathOperator*{{\\argmin}}{{arg\\,min}}

%% The hyperref package for clickable links in PDF and also for storing
%% metadata to PDF (including the table of contents).
%% Most settings are pre-set by the pdfx package.
\\hypersetup{{unicode}}
\\hypersetup{{breaklinks=true}}

% Definitions of macros (see description inside)
\\include{{macros}}
\\begin{{document}}
\\begin{{figure*}}[h!]
\\centering
\\scalebox{{0.75}}{{
\\begin{{tikzpicture}}


\\node(tables)[draw, inner sep=5pt, rounded corners, text width=39em]{{
    \\small
    \\begin{{center}}
        \\begin{{tabular}}{{llccccccc}}
        \\toprule
        CITY & TEAM & WIN & LOSS & PTS$_1$ & FG\\%$_2$ & REB$_3$ & AST$_4$ & FG3\\% \\\\
        \\midrule
        {TEAM-CITY_1} & {TEAM-NAME_1} & {TEAM-WINS_1}  & {TEAM-LOSSES_1}    & {TEAM-PTS_1} & {TEAM-FG_PCT_1}& {TEAM-REB_1} & {TEAM-AST_1} & {TEAM-FG3_PCT_1} \\\\
        {TEAM-CITY_2} & {TEAM-NAME_2}  & {TEAM-WINS_2}   & {TEAM-LOSSES_2}   & {TEAM-PTS_2} & {TEAM-FG_PCT_2}  & {TEAM-REB_2} & {TEAM-AST_2} & {TEAM-FG3_PCT_2} \\\\
        \\bottomrule
        \\end{{tabular}}
        \\vspace{{1.0cm}}
        \\begin{{tabular}}{{lccccccc}}
        \\toprule
        TEAM          & PTS\\_QTR1      & PTS\\_QTR2   & PTS\\_QTR3      & PTS\\_QTR4      & FT\\% & TOV \\\\
        \\midrule
        {TEAM-NAME_1}  & {TEAM-PTS_QTR1_1} & {TEAM-PTS_QTR2_1} & {TEAM-PTS_QTR3_1} & {TEAM-PTS_QTR4_1}    & {TEAM-FT_PCT_1}  & {TEAM-TOV_1} \\\\
        {TEAM-NAME_2} & {TEAM-PTS_QTR1_2} & {TEAM-PTS_QTR2_2}  & {TEAM-PTS_QTR3_2} & {TEAM-PTS_QTR4_2}  & {TEAM-FT_PCT_2} & {TEAM-TOV_2} \\\\
        \\bottomrule
        \\end{{tabular}}
        \\vspace{{0.5cm}}

        \\begin{{tabular}}{{llllllll}}
            \\toprule
            PLAYER  & City         & PTS$_1$ & AST$_4$ & REB$_3$ & FG$_5$  & FGA$_6$ & S\\_POS$_7$ $\\ldots$ \\\\
            \\midrule
            {PLAYER_NAME_1} & {TEAM_CITY_1} & {PTS_1} & {AST_1} & {REB_1} & {FGM_1} & {FGA_1} & {START_POSITION_1} $\\ldots$ \\\\
            {PLAYER_NAME_2} & {TEAM_CITY_2} & {PTS_2} & {AST_2} & {REB_2} & {FGM_2} & {FGA_2} & {START_POSITION_2} $\\ldots$ \\\\
            {PLAYER_NAME_3} & {TEAM_CITY_3} & {PTS_3} & {AST_3} & {REB_3} & {FGM_3} & {FGA_3} & {START_POSITION_3} $\\ldots$ \\\\
            {PLAYER_NAME_4} & {TEAM_CITY_4} & {PTS_4} & {AST_4} & {REB_4} & {FGM_4} & {FGA_4} & {START_POSITION_4} $\\ldots$ \\\\
            {PLAYER_NAME_5} & {TEAM_CITY_5} & {PTS_5} & {AST_5} & {REB_5} & {FGM_5} & {FGA_5} & {START_POSITION_5} $\\ldots$ \\\\
            {PLAYER_NAME_6} & {TEAM_CITY_6} & {PTS_6} & {AST_6} & {REB_6} & {FGM_6} & {FGA_6} & {START_POSITION_6} $\\ldots$ \\\\
            {PLAYER_NAME_7} & {TEAM_CITY_7} & {PTS_7} & {AST_7} & {REB_7} & {FGM_7} & {FGA_7} & {START_POSITION_7} $\\ldots$ \\\\
            {PLAYER_NAME_8} & {TEAM_CITY_8} & {PTS_8} & {AST_8} & {REB_8} & {FGM_8} & {FGA_8} & {START_POSITION_8} $\\ldots$ \\\\
            {PLAYER_NAME_9} & {TEAM_CITY_9} & {PTS_9} & {AST_9} & {REB_9} & {FGM_9} & {FGA_9} & {START_POSITION_9} $\\ldots$ \\\\
            {PLAYER_NAME_10} & {TEAM_CITY_10} & {PTS_10} & {AST_10} & {REB_10} & {FGM_10} & {FGA_10} & {START_POSITION_10} $\\ldots$ \\\\
            {PLAYER_NAME_11} & {TEAM_CITY_11} & {PTS_11} & {AST_11} & {REB_11} & {FGM_11} & {FGA_11} & {START_POSITION_11} $\\ldots$ \\\\
            {PLAYER_NAME_12} & {TEAM_CITY_12} & {PTS_12} & {AST_12} & {REB_12} & {FGM_12} & {FGA_12} & {START_POSITION_12} $\\ldots$ \\\\
            {PLAYER_NAME_13} & {TEAM_CITY_13} & {PTS_13} & {AST_13} & {REB_13} & {FGM_13} & {FGA_13} & {START_POSITION_13} $\\ldots$ \\\\
            \\ldots
        \\end{{tabular}}
    \\end{{center}}
}}; % end of node
\\node(summary) [rectangle, draw,thick,fill=blue!0,text width=39em, rounded corners, inner sep =8pt, minimum height=1em, below=-2mm of tables]{{
    \\baselineskip=100pt
    \\small
    {summary}
    \\par
}};
\\node[rectangle, below=2mm of summary, text width=40em] {{
    \\footnotesize \\textit{{Note:}} $_1$ Points; $_2$ Field Goal Percentage; $_3$ Rebounds; $_4$ Assists; $_5$ Field Goals; $_6$ Field Goals Attempted; $_7$ Starting Position; $N/A$ means undefined value
}};
\\end{{tikzpicture}}
}}
\\end{{figure*}}
\\end{{document}}"""

def create_latex_table_sum( data
                          , summary
                          , max_table_size
                          , ix_to_tk
                          , ix_to_tp
                          , out_path):
    _, types, entities, values, has = data
    types, entities, values, has = types[0].numpy(), entities[0].numpy(), values[0].numpy(), has[0].numpy()

    entity_span = 22
    team_span = 15
    ix = 2
    entities = []
    teams = []
    while ix < max_table_size:
        entity = {}
        try:
            LineScoreEntries(ix_to_tp[types[ix]])
            team = {}
            for _ in range(2):
                for k in range(team_span):
                    value = " ".join(ix_to_tk[values[ix+k]].split("_"))
                    type = ix_to_tp[types[ix+k]]
                    team[type] = value
                teams.append(team)
                team = {}
                ix += team_span
            break
        except:
            for k in range(entity_span):
                type = ix_to_tp[types[ix+k]]
                if type in [ "PLAYER_NAME", "START_POSITION", "TEAM_CITY"]:
                    value = " ".join(ix_to_tk[values[ix+k]].split("_"))
                else:
                    try:
                        value = int(ix_to_tk[values[ix+k]])
                    except:
                        value = 0
                entity[type] = value
            entities.append(entity)
            ix += entity_span
    format_dict = {}
    entities.sort(key=lambda x: x["PTS"], reverse=True)
    for lst in teams, entities:
        for ix, ent in enumerate(lst):
            for type, value in ent.items():
                format_dict[f"{type}_{ix+1}"] = value
    format_dict["summary"] = summary
    with open(out_path, 'w', encoding="utf8") as f:
        print(figure_template.format(**format_dict), file=f)

def main(args):
    config_path = os.path.join(args.dataset_path, "config.txt")
    test_path = os.path.join(args.dataset_path, "test.tfrecord")
    valid_path = os.path.join(args.dataset_path, "valid.tfrecord")
    vocab_path = os.path.join(args.dataset_path, "all_vocab.txt")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    max_table_size, max_summary_size, max_cp_size = \
        load_values_from_config( config_path, load_cp=False)
    batch_size = 1

    if args.dataset == "test":
        path = test_path
        if not args.gold:
          sum_path = os.path.join(args.summaries_path, "test_preds.txt")
        else:
          sum_path = os.path.join(args.summaries_path, "test_golds.txt")
    elif args.dataset == "valid":
        path = valid_path
        if not args.gold:
          sum_path = os.path.join(args.summaries_path, "val_preds.txt")
        else:
          sum_path = os.path.join(args.summaries_path, "val_golds.txt")
    else:
        raise RuntimeError("Invalid name of the dataset! (only test | valid are allowed)")
    
    dataset, _, tk_to_ix, tp_to_ix, ha_to_ix, pad, bos, eos \
            = load_tf_record_dataset( path=path
                                    , vocab_path=vocab_path
                                    , batch_size=batch_size
                                    , shuffle=False
                                    , preprocess_table_size=max_table_size
                                    , preprocess_summary_size=max_summary_size)
    
    ix_to_tk = dict([(value, key) for key, value in tk_to_ix.items()])
    ix_to_tp = dict([(value, key) for key, value in tp_to_ix.items()])

    with open(sum_path, 'r', encoding="utf8") as f:
        summaries = f.read().strip().split("\n")

    for ix, datapoint in enumerate(dataset):
        if ix != args.example_number:
            continue
        else:
            create_latex_table_sum( datapoint
                                  , summaries[ix]
                                  , max_table_size
                                  , ix_to_tk
                                  , ix_to_tp
                                  , args.out_path)
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("example_number", type=int)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("summaries_path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--gold", action='store_true')
    main(parser.parse_args())
