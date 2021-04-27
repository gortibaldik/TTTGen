from argparse import ArgumentParser

"""
Script used for retrieving the stats used in the thesis - rotowire train dataset token stats, and overlap of valid and test dataset stats

- the data for this script are generated from ~/rotowire/preprocessing/preprecessing.py <rotowire_dir> extract_summaries <output_dir> with one tweak, all the name_transformations in the script are commented out
"""

def main(args):
  unique_train_tokens = {}
  with open(args.train, 'r', encoding='utf8') as f:
    lines = f.read().strip().split('\n')
    for line in lines:
      words = line.strip().split()
      for word in words:
        unique_train_tokens[word] = unique_train_tokens.get(word, 0) + 1
    
    print("---")
    print(f"unique tokens : {len(unique_train_tokens)}")
    more_than_5 = dict([(key, value) for key, value in unique_train_tokens.items() if value >= 5])
    print(f"tokens with more than 5 occurrences : {len(more_than_5)}")
    print(f"relative more than 5 : {len(more_than_5)*100/ len(unique_train_tokens)}")

  for name in [args.train, args.valid, args.test]:
    unique_tokens = {}
    with open(name, 'r', encoding='utf8') as f:
      lines = f.read().strip().split('\n')
      for line in lines:
        words = line.strip().split()
        for word in words:
          unique_tokens[word] = unique_tokens.get(word, 0) + 1
      
      print("---")
      print(f"unique tokens in {name} : {len(unique_tokens)}")
      overlap_dict = dict([(key, value) for key, value in unique_tokens.items() if key in unique_train_tokens])
      mt5 = dict([(key, value) for key, value in unique_tokens.items() if key in more_than_5])
      print(f"relative overlap with train of {name} : {len(overlap_dict)*100/len(unique_tokens)}")
      print(f"relative overlap with train_mt5 of {name} : {len(mt5)*100/len(unique_tokens)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train", type=str)
    parser.add_argument("valid", type=str)
    parser.add_argument("test", type=str)
    main(parser.parse_args())
