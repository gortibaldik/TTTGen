set -e

print_info() {
  file=$1
  echo -n "total number of tokens: "
  wc -l "${file}" | cut -f1 -d' '
  echo -n "number of tokens with more than five occurrences: "
  grep -nh '5$' "${file}" | tail -1 | cut -f1 -d:
}

format=$4
rotowire_dir=$3
out_dir=$2
num_merges=$1

DELETE_VOCAB=0
# if codes for bpe weren't learnt yet
if [ ! -f "${out_dir}/codes_train_${num_merges}.txt" ]; then
  echo "codes weren't learnt yet, gonna create them in ${out_dir}"
  ./learn_bpe_codes.sh "${num_merges}" "${out_dir}" "${rotowire_dir}"
  DELETE_VOCAB=1
  echo -e "---\n"
fi

# pfa means << Prepared For Application of byte pair encoding >>
# creates ${out_dir}/{train, valid, test}_pfa.txt if they don't exist yet
for f in "train" "valid" "test"
do
  if [ ! -f "${out_dir}/${f}_pfa.txt" ]; then
    echo "preparing the summaries for bpe application (${out_dir}/${f}_pfa.txt)"
    python3 preprocessing.py "${rotowire_dir}" --only_set="${f}" \
                                               extract_summaries \
                                               --output_dir="${out_dir}" \
                                               --transform_players \
                                               --prepare_for_bpe_application
  fi
done

# create ${out_dir}/{train, valid, test}_prepared.txt
# files segmented in right way, prepared as the input for neural net
# also create vocabs
for f in "train" "valid" "test"
do
  echo "applying bpe to ${f} dataset"
  subword-nmt apply-bpe -c "${out_dir}/codes_train_${num_merges}.txt" \
                        -i "${out_dir}/${f}_pfa.txt" \
                        --glossaries "<<<[^>]*>>>" "[0-9]+"| 
                        sed -r 's/<<<([^>]*)>>>/\1/g' > "${out_dir}/${f}_prepared.txt"

  subword-nmt get-vocab -o "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt" \
                        -i "${out_dir}/${f}_prepared.txt"
  print_info "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt"
done

echo -e "---\ncreating global vocab"
cat "${out_dir}/train_prepared.txt" \
    "${out_dir}/valid_prepared.txt" \
    "${out_dir}/test_prepared.txt" |
    subword-nmt get-vocab > "${out_dir}/tmp_vocab.txt"

print_info "${out_dir}/tmp_vocab.txt"

subword-nmt get-vocab -i "${out_dir}/train_prepared.txt" -o "${out_dir}/token_vocab.txt"

# store biggest wc to config
echo "collecting biggest summary stat"

# do not modify config.txt if it already contains all the needed data
file_name="${out_dir}/config.txt"
if [ $(wc -l "${out_dir}/config.txt" | cut -f1 -d' ') -eq 2 ]; then
  file_name="/dev/null"
fi

python3 word_count.py --log \
                      "${out_dir}/train_prepared.txt" \
                      "${out_dir}/valid_prepared.txt" \
                      "${out_dir}/test_prepared.txt" >> "${file_name}"

# create tfrecord dataset
echo "creating ${format} dataset"
dataset_dir="${out_dir}_${format}"
if [ ! -d "${dataset_dir}" ]; then
  mkdir "${dataset_dir}"
  if [ "$format" == "tfrecord" ]; then
    python3 preprocessing.py "${rotowire_dir}" --log \
                                               create_dataset \
                                               --preproc_summaries_dir="${out_dir}" \
                                               --output_dir="${dataset_dir}" \
                                               --to_tfrecord
  elif [ "$format" == "txt" ]; then
    python3 preprocessing.py "${rotowire_dir}" --log \ 
                                               create_dataset \
                                               --preproc_summaries_dir="${out_dir}" \
                                               --output_dir="${dataset_dir}" \
                                               --to_txt
  elif [ "$format" == "np" ]; then
    python3 preprocessing.py "${rotowire_dir}" --log 
                                               create_dataset \
                                               --preproc_summaries_dir="${out_dir}" \
                                               --output_dir="${dataset_dir}" \
                                               --to_npy
  else
    echo "Invalid format specified!"
  fi
 cp "${out_dir}/config.txt" "${dataset_dir}/config.txt"
fi

echo "cleaning"
rm "${out_dir}/tmp_vocab.txt"
for f in "train" "valid" "test"
do
  rm "${out_dir}/${f}_pfa.txt"
  rm "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt"
done
echo "done"
