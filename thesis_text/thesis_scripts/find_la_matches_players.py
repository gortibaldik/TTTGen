from argparse import ArgumentParser
import json

cl = ["Clippers", "Lakers"]

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("file", type=str)
  args = parser.parse_args()
  with open(args.file, 'r', encoding='utf8') as f:
    for ix, match in enumerate(json.load(f)):
      if match["home_name"] in cl and match["vis_name"] in cl:
        print(f"---{ix}")
        for r in match["box_score"]["TEAM_CITY"].items():
          print(f"{r[0]} : {r[1]}")
        input()
