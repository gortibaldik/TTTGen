from argparse import ArgumentParser
import nltk.tokenize as nltk_tok


def main(args):
  with open(args.file, 'r', encoding="utf8") as f:
    in_summaries = f.read().strip().split('\n')

  out_summaries = []
  for s in in_summaries:
   sentences = nltk_tok.sent_tokenize(s)
   out_sentences = []
   for sentence in sentences:
     out_sentences.append(sentence[0].upper() + sentence[1:])
   out_summaries.append(" ".join(out_sentences))
  
  with open(args.file, 'w', encoding="utf8") as f:
    f.write("\n".join(out_summaries))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("file", type=str)
  main(parser.parse_args())
