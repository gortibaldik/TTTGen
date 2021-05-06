#!/usr/bin/env python3

from argparse import ArgumentParser

TABLE=False
SWITCH=False

def table_main(args):
    wd = {}
    for name in args.file_names:
        with open(name, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                words = line.strip().split()
                for word in words:
                    if SWITCH:
                      wd[word] = wd.get(word, 0) + 1
                    else:
                      kss = word.split(':')
                      if len(kss) > 2:
                        continue
                      kss = kss[0].split("_")
                      if len(kss) < 2:
                        continue
                      field = "".join(kss[:-1])
                      wd[field] = wd.get(field, 0) + 1
        print(f"number of unique words in {name} : {len(wd)}")
        print(f"total number of words : {sum([value for value in wd.values()])}")
        print(f"total number of words with occurrence at least 100 : {len([value for value in wd.values() if value >= 100])}")


def sum_main(args):
    for name in args.file_names:
        less_than = [ [25,0], [50,0], [75,0], [100,0], [125,0]]
        sum_len = 0
        with open(name, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                words = line.strip().split()
                sum_len += len(words)
                for arr in less_than:
                    if len(words) <= arr[0]:
                        arr[1] += 1
        for arr in less_than:
            print(f"<={arr[0]} : {arr[1]} : {arr[1] / len(lines)}")
        print(f"avg_len : {sum_len / len(lines)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('file_names', metavar='FILE_NAME', type=str, nargs='+')
    if TABLE:
      table_main(parser.parse_args())
    else:
      sum_main(parser.parse_args())
