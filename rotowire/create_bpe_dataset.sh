set -e

rotowire_dir=$3
out_dir=$2
num_merges=$1

DELETE_VOCAB=0
# if dictionaries weren't created yet, create them
if [ ! -f "${out_dir}/all_pfbpe_vocab_${num_merges}.txt" ]; then
  echo "vocabs weren't created yet, gonna create them in ${out_dir}"
  ./create_bpe_vocab.sh "${num_merges}" "${out_dir}" "${rotowire_dir}"
  DELETE_VOCAB=1
  echo -e "---\n"
fi

# pfa means << Prepared For Application of byte pair encoding >>
# creates ${out_dir}/{train, valid, test}_pfa.txt if they don't exist yet
for f in "train" "valid" "test"
do
  if [ ! -f "${out_dir}/${f}_pfa.txt" ]; then
    echo "preparing the summaries for bpe application"
    python3 preprocessing.py "${rotowire_dir}" extract_summaries \
                                               --output_dir="${out_dir}" \
                                               --transform_players \
                                               --prepare_for_bpe_application
    break
  fi
done

# create ${out_dir}/{train, valid, test}_prepared.txt
# files segmented in right way, prepared as the input for neural net
for f in "train" "valid" "test"
do
  echo "applying bpe to ${f} dataset"
  subword-nmt apply-bpe -c "${out_dir}/codes_train_${num_merges}.txt" \
                        -i "${out_dir}/${f}_pfa.txt" \
                        --glossaries="<<<[^>]*>>>" | 
                        sed -r 's/<<<([^>]*)>>>/\1/g' > "${out_dir}/${f}_prepared.txt"
done

# store biggest wc to config
echo "collecting biggest summary stat"
python3 word_count.py "${out_dir}/train_prepared.txt" \
                                       "${out_dir}/valid_prepared.txt" \
                                       "${out_dir}/test_prepared.txt" >> "${out_dir}/config.txt"
echo "cleaning"
# cleaning
for f in "train" "valid" "test"
do
  if [ $DELETE_VOCAB -eq 1 ]; then
    rm "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt"
  fi
  rm "${out_dir}/${f}_pfa.txt"
done

if [ $DELETE_VOCAB -eq 1 ]; then
  mv "${out_dir}/all_pfbpe_vocab_${num_merges}.txt" \
     "${out_dir}/token_vocab.txt"
fi

echo "done"
