from argparse import ArgumentParser
from summary_class import Summary
from utils import OccurrenceDict

def main(args):
    ss = []
    with open(args.file, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            wd = OccurrenceDict()
            ss.append(Summary(line.strip().split(), wd))
    
    with open(args.out_path, 'w', encoding='utf8') as f:
        for s in ss:
            print(str(s), file=f)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("out_path", type=str)
    main(parser.parse_args())