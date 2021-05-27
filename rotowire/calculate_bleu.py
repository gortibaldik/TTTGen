#!/usr/bin/env python3
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from argparse import ArgumentParser

def _main(args):
    if args.sf == 1:
      sf = SmoothingFunction().method1
    elif args.sf == 2:
      sf = SmoothingFunction().method2
    elif args.sf == 3:
      sf = SmoothingFunction().method3
    else:
      sf = SmoothingFunction().method4

    with open(args.predicted, 'r') as f:
        predicted = [ p.strip().split() for p in f.read().strip().split('\n') ]
    with open(args.gold, 'r') as f:
        gold = [ g.strip().split() for g in f.read().strip().split('\n')]
    
    sum_bleu = 0
    for ix, _ in enumerate(predicted):
        bleu_score = sentence_bleu([gold[ix]], predicted[ix], smoothing_function=sf)
        sum_bleu += bleu_score
    
    print(f"macro averaged bleu over the inputs: {sum_bleu / len(predicted)}")

    gold_corpus = [[g] for g in gold]
    # print(" ".join(predicted[0]))
    # print(" ".join(gold_corpus[0][0]))
    print(f"micro-averaged bleu over the inputs: {corpus_bleu(gold_corpus, predicted,  smoothing_function=sf)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("predicted", type=str)
    parser.add_argument("gold", type=str)
    parser.add_argument("--sf", type=int, help="Smoothing Function to be used", default=4)
    _main(parser.parse_args())
