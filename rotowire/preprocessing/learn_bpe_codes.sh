set -e

prun_records=$5
advanced_transformations=$4
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

if [ ! -f "${out_dir}/train_pfa.txt" ]; then

  echo "${out_dir}/train_pfa.txt doesn't exist, it is going to be created!"
  script="python3 preprocessing.py \"${rotowire_dir}\" \
                                    --only_train \
                                    extract_summaries \
                                    --words_limit=900 \
                                    --output_dir=\"${out_dir}\" \
                                    --prepare_for_bpe_application \
                                    --entity_vocab_path=\"${out_dir}/entity_vocab.txt\" \
                                    --cell_vocab_path=\"${out_dir}/cell_vocab.txt\" \
                                    --config_path=\"${out_dir}/config.txt\""
  if [ ! -z "$advanced_transformations" ]; then
    echo "pfa and pfbpe with advanced transformations"
    script="${script} --transform_players \
                      --lowercase \
                      --exception_cities \
                      --exception_teams"
  else
    echo "pfa and pfbpe train basic"
  fi

  if [ ! -z "$prun_records" ]; then
    echo "prunning records"
    script="${script} --prun_records \
                      --order_records"
  fi

  eval "${script}"
  # as the input for learn-bpe the text should be without any
  # tokens not desirable to be merged, therefore each player_token (which is 
  # enclosed in <<<player_token>>> after preprocessing) is gonna be removed
  # also each number token should be removed
  sed 's/<<<[^>]*>>>//g;s/[0-9]+//g' "${out_dir}/train_pfa.txt" > "${out_dir}/train_pfbpe.txt"

elif [ ! -f "${out_dir}/train_pfbpe.txt" ]; then

  # ${out_dir}/train_pfa.txt must exist
  sed 's/<<<[^>]*>>>//g;s/[0-9]+//g' "${out_dir}/train_pfa.txt" > "${out_dir}/train_pfbpe.txt"
fi

# learn bpe codes file in ${out_dir}/codes_train_${num_merges}.txt
echo "learning bpe!"
subword-nmt learn-bpe -s "${num_merges}" \
                      -o "${out_dir}/codes_train_${num_merges}.txt" \
                      -i "${out_dir}/train_pfbpe.txt"

# cleaning
rm "${out_dir}/train_pfbpe.txt"

# after the script, files: train_pfa.txt
#                          codes_train_${num_merges}.txt
#                          entity_vocab.txt
#                          cell_vocab.txt
# should be present in the ${out_dir}
