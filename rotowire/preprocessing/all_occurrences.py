#!/usr/bin/env python3

from argparse import ArgumentParser
import sys

def find_tokens(file_path, searched : str,before : bool, after : bool):
    with open(file_path, 'r') as f:
        content = f.read().strip().split('\n')
    for line_num, line in enumerate(content):
      tokens = line.strip().split()
      ix = 0
      for token in tokens:
        if searched in token:
          print(f"{file_path} : {line_num}: ", end="")
          if before and ix > 0:
            print(tokens[ix-1], end=" ")
          print(token, end="")
          if after and ix < len(tokens)-1:
            print(f" {tokens[ix+1]}", end="")
          print()
        ix += 1

if __name__ == "__main__":
  parser = ArgumentParser(
    description='Print length of the line with most words from all the files provided'
  )
  parser.add_argument('file_names', metavar='FILE_NAME', type=str, nargs='+')
  parser.add_argument('--searched', required=True, type=str)
  parser.add_argument('--before', help='print token before searched', action='store_true')
  parser.add_argument('--after', help='print token after searched', action='store_true')
  args = vars(parser.parse_args())

  wc = 0
  for file_path in args['file_names']:
    t = find_tokens(file_path, args['searched'], args['before'], args['after'])

