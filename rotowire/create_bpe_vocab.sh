set -e

print_info() {
  file=$1
  echo -n "total number of tokens: "
  wc -l "${file}" | cut -f1 -d' '
  echo -n "number of tokens with more than five occurrences: "
  grep -nh '5$' "${file}" | tail -1 | cut -f1 -d:
}

rotowire_dir=$3
out_dir=$2
num_merges=$1

if [ ! -d "${rotowire_dir}" ]; then
  1>&2 echo "directory with .json rotowire files (${rotowire_dir}) doesn't exist, quitting" 
fi

# pfa means <<<Preapared For Application of byte pair encoding>>>
# pfbpe mean <<<Prepared For Byte Pair Encoding>>> -- training
# create directory ${out_dir} and files ${out_dir}/{train, valid, test}_pfa.txt
# plus file with all the player tokens from concatenation of all the summaries
# in ${out_dir}/player_vocab.txt
if [ ! -d "$out_dir" ]; then
  echo "directory ${out_dir} doesn't exist, creating ${out_dir}"
  mkdir ${out_dir}
fi

for f in "train" "valid" "test"
do
  if [ ! -f "${out_dir}/${f}_pfa.txt" ]; then

    echo "${out_dir}/${f}_pfa.txt doesn't exist, it is going to be created!"
    python3 preprocessing.py "${rotowire_dir}" extract_summaries \
                                               --output_dir="${out_dir}" \
                                               --transform_players \
                                               --prepare_for_bpe_application \
                                               --entity_vocab_path="${out_dir}/entity_vocab.txt" \
                                               --cell_vocab_path="${out_dir}/cell_vocab.txt"
    # as the input for learn-bpe the text should be without any
    # tokens not desirable to be merged, therefore each player_token (which is 
    # enclosed in <<<player_token>>> after preprocessing) is gonna be removed
    sed 's/<<<[^>]*>>>//g' "${out_dir}/${f}_pfa.txt" > "${out_dir}/${f}_pfbpe.txt"

  elif [ ! -f "${out_dir}/${f}_pfbpe.txt" ]; then

    # ${out_dir}/${f}_pfa.txt must exist
    sed 's/<<<[^>]*>>>//g' "${out_dir}/${f}_pfa.txt" > "${out_dir}/${f}_pfbpe.txt"
  fi
done

# learn bpe codes file in ${out_dir}/codes_train_${num_merges}.txt
echo "learning bpe!"
subword-nmt learn-bpe -s "${num_merges}" \
                      -o "${out_dir}/codes_train_${num_merges}.txt" \
                      -i "${out_dir}/train_pfbpe.txt"

# apply learned bpe to ${out_dir}/{train, valid, test, all}_pfbpe.txt
# and extract the vocab from those files
for f in "train" "valid" "test"
do
  echo -e "---\ncreating ${f} vocab"
  subword-nmt apply-bpe -c "${out_dir}/codes_train_${num_merges}.txt" \
                        -i "${out_dir}/${f}_pfbpe.txt" \
                        -o "${out_dir}/${f}_encoded_${num_merges}.txt"

  subword-nmt get-vocab -o "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt" \
                        -i "${out_dir}/${f}_encoded_${num_merges}.txt"
  print_info "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt"
done

echo -e "---\ncreating global vocab"
cat "${out_dir}/train_encoded_${num_merges}.txt" \
    "${out_dir}/valid_encoded_${num_merges}.txt" \
    "${out_dir}/test_encoded_${num_merges}.txt" |
    subword-nmt get-vocab -o "${out_dir}/all_pfbpe_vocab_${num_merges}.txt"

print_info "${out_dir}/all_pfbpe_vocab_${num_merges}.txt"

# cleaning
for f in "train" "valid" "test"
do
  rm "${out_dir}/${f}_encoded_${num_merges}.txt"
  rm "${out_dir}/${f}_pfbpe.txt"
done

# after the script, files: {train, valid, test, all}_pfbpe_vocab_${num_merges}.txt;
#                          {train, valid, test}_pfa.txt
#                          {train, valid, test}_pfbpe.txt
#                          codes_train_${num_merges}.txt
#                          entity_vocab.txt
#                          cell_vocab.txt
# should be present in the ${out_dir}
