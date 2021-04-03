from argparse import ArgumentParser

def get_most_words_line_count(file_path, log):
    with open(file_path, 'r') as f:
        content = f.read().strip().split('\n')
    m = 0
    sm = 0
    for line in content:
        ln = len(line.strip().split())
        if ln > m: m = ln
        sm += ln
    if log:
      print(f"{file_path} : max : {m} average : {sm / len(content)}")
    return m

if __name__ == "__main__":
  parser = ArgumentParser(
    description='Print length of the line with most words from all the files provided'
  )
  parser.add_argument('file_names', metavar='FILE_NAME', type=str, nargs='+')
  parser.add_argument('--log', help='print stats of individual files', action='store_true')
  args = vars(parser.parse_args())

  wc = 0
  for file_path in args['file_names']:
    t = get_most_words_line_count(file_path, args['log'])
    if wc < t: wc = t

  print(wc)

