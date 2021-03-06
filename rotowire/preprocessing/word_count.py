#!/usr/bin/env python3

from argparse import ArgumentParser
import sys

def get_most_words_line_count(file_path, log):
    """ Traverse file and compute various statistics about word-length of its lines

    Args:
      file_path : path to file
      log:  whether to show minimal, maximal and average length of lines in the file
    
    Returns:
      maximal length of line in the file
    """
    with open(file_path, 'r') as f:
        content = f.read().strip().split('\n')
    _max = 0
    _min = 1000000  # just some ridiculously high number
    sm = 0
    for line in content:
        ln = len(line.strip().split())
        if ln > _max: _max = ln
        if ln < _min: _min = ln
        sm += ln
    if log:
      print(f"{file_path} : max : {_max} min : {_min} average : {sm / len(content)}", file=sys.stderr)
    return _max

if __name__ == "__main__":
  parser = ArgumentParser( description='Print length of the line with most words from all the files provided')
  parser.add_argument( 'file_names'
                     , metavar='FILE_NAME'
                     , type=str, nargs='+')
  parser.add_argument( '--log'
                     , help='print stats of individual files'
                     , action='store_true')
  args = vars(parser.parse_args())

  wc = 0
  for file_path in args['file_names']:
    t = get_most_words_line_count(file_path, args['log'])
    if wc < t: wc = t
      
  print(wc)

