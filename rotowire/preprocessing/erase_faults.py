"""
This script is used to erase specific entries from the rotowire dataset. Cleaning needs to be
performed as these entries harm the overall performance of the model on the dataset.

// we expect the original dataset downloaded from https://github.com/harvardnlp/boxscore-data
"""

from argparse import ArgumentParser
import json
import os

entries_numbers = { 'train.json': [
    36, 84, 243, 280, 403, 603, 822, 1079, 1179, 1499, 1504, 1515, 1767, 1815, 1849, 1917, 2117,
    2296, 2306, 2312, 2379, 2552, 2688, 2708, 2797, 2959, 3307, 3391
  ],
  'valid.json': [
    120, 177, 186, 356, 386, 597
  ],
  'test.json': [516]
}

def main(args):
  for key in entries_numbers.keys():
    path_in = os.path.join(args.dir_in, key)
    path_out = os.path.join(args.dir_out, key)
    print(f"{path_in} -> {path_out}")
    to_be_erased = entries_numbers[key]
    print(f"to_be_erased : {to_be_erased} ")
    matches = []
    with open(path_in, 'r', encoding='utf8') as f:
      for ix, match in enumerate(json.load(f)):
        if (ix + 1) in entries_numbers[key]:
          continue
        else:
          matches.append(match)
    with open(path_out, 'w', encoding='utf8') as f:
      json.dump(matches, f)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("dir_in", type=str)
  parser.add_argument("dir_out", type=str)
  main(parser.parse_args())
