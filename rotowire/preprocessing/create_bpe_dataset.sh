set -e

print_info() {
  file=$1
  echo -n "total number of tokens: "
  wc -l "${file}" | cut -f1 -d' '
  echo -n "number of tokens with more than five occurrences: "
  grep -nh '5$' "${file}" | tail -1 | cut -f1 -d:
}
# initialization of all the settable attributes to ""
# just to be sure they aren't set
content_plan=""
order_records=""
prun_records=""
format=""
npy=""
to_txt=""
tfrecord=""
advanced_transformations=""

# positional arguments
rotowire_dir=$3
out_dir=$2
num_merges=$1

# parse optional arguments
while :; do
    case $4 in
        --tfrecord) tfrecord="SET"
                    format="tfrecord"
        ;;
        --npy) npy="SET"
               format="npy"
        ;;
        --txt) to_txt="SET"
               format="txt"
        ;;
        --adv) advanced_transformations="SET"
        ;;
        --content_plan) content_plan="SET"
        ;;
        --order_records) order_records="SET"
        ;;
        --prun_records) prun_records="SET"
                        order_records="SET"
        ;;
        *) break
    esac
    shift
done

DELETE_VOCAB=0
# if codes for bpe weren't learnt yet
if [ ! -f "${out_dir}/codes_train_${num_merges}.txt" ]; then
  echo "codes weren't learnt yet, gonna create them in ${out_dir}"
  ./learn_bpe_codes.sh "${num_merges}" \
                      "${out_dir}" \
                      "${rotowire_dir}" \
                      "${advanced_transformations}" \
                      "${prun_records}"
  DELETE_VOCAB=1
  echo -e "---\n"
fi

# pfa means << Prepared For Application of byte pair encoding >>
# creates ${out_dir}/{train, valid, test}_pfa.txt if they don't exist yet
for f in "train" "valid" "test"
do
  if [ ! -f "${out_dir}/${f}_pfa.txt" ]; then
    echo "preparing the summaries for bpe application (${out_dir}/${f}_pfa.txt)"
    if [ -z "$advanced_transformations" ]; then
      echo "preparing ${f}_bpa basic"
      python3 preprocessing.py "${rotowire_dir}" --only_set="${f}" \
                                                extract_summaries \
                                                --words_limit=900 \
                                                --output_dir="${out_dir}" \
                                                --transform_players \
                                                --prepare_for_bpe_application
    else
      echo "preparing ${f}_bpa with advanced transformations"
      python3 preprocessing.py "${rotowire_dir}" --only_set="${f}" \
                                                 extract_summaries \
                                                 --words_limit=900 \
                                                 --output_dir="${out_dir}" \
                                                 --transform_players \
                                                 --prepare_for_bpe_application \
                                                 --lowercase \
                                                 --exception_cities \
                                                 --exception_teams
    fi
  fi
done

# at first apply bpe to train dataset, get vocab
subword-nmt apply-bpe -c "${out_dir}/codes_train_${num_merges}.txt" \
                      -i "${out_dir}/train_pfa.txt" \
                      --glossaries "<<<[^>]*>>>|[0-9]+" |
                      sed -r 's/<<<([^>]*)>>>/\1/g' > "${out_dir}/train_prepared.txt"

subword-nmt get-vocab -o "${out_dir}/train_pfbpe_vocab_${num_merges}.txt" \
                      -i "${out_dir}/train_prepared.txt"
print_info "${out_dir}/train_pfbpe_vocab_${num_merges}.txt"

# create ${out_dir}/{valid, test}_prepared.txt
# files segmented in such a way, that apply-bpe would produce only subwords appearing in the 
# train set, therefore no OOV will be produced
for f in "valid" "test"
do
  echo "applying bpe to ${f} dataset"
  subword-nmt apply-bpe -c "${out_dir}/codes_train_${num_merges}.txt" \
                        -i "${out_dir}/${f}_pfa.txt" \
                        --vocabulary "${out_dir}/train_pfbpe_vocab_${num_merges}.txt" \
                        --vocabulary-threshold 2 \
                        --glossaries "<<<[^>]*>>>|[0-9]+" |
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
if [ $(wc -l "${out_dir}/config.txt" | cut -f1 -d' ') -ge 2 ]; then
  file_name="/dev/null"
fi

python3 word_count.py --log \
                      "${out_dir}/train_prepared.txt" \
                      "${out_dir}/valid_prepared.txt" \
                      "${out_dir}/test_prepared.txt" >> "${file_name}"

# create dataset in specified format
echo "creating ${format} dataset"
dataset_dir="${out_dir}_${format}"
script="python3 preprocessing.py \"${rotowire_dir}\" --log \
        create_dataset \
        --preproc_summaries_dir=\"${out_dir}\""

if [ "$format" == "tfrecord" ]; then
  script="${script} --to_tfrecord"
elif [ "$format" == "txt" ]; then
  script="${script} --to_txt"
elif [ "$format" == "npy" ]; then
  script="${script} --to_npy"
else
  echo "Invalid format"
fi

if [ ! -z "${content_plan}" ]; then
  # check if content plans are present
  if [ ! -d "${rotowire_dir}/content_plans" ]; then
    echo "${rotowire_dir}/content_plans must be part of the dataset to be able to create dataset with it"
    exit 1
  fi
  script="${script} --content_plans_dir=\"${rotowire_dir}/content_plans\""
  dataset_dir="${dataset_dir}_cp"

  # append length information to config file
  file_name="${out_dir}/config.txt"
  if [ $(wc -l "${out_dir}/config.txt" | cut -f1 -d' ') -ge 3 ]; then
    file_name="/dev/null"
  fi
  python3 word_count.py "${rotowire_dir}/content_plans/train.txt" \
                        "${rotowire_dir}/content_plans/valid.txt" >> "${file_name}"
fi

if [ ! -z "${order_records}" ]; then
  script="${script} --order_records"
fi

if [ ! -z "${prun_records}" ]; then
  script="${script} --prun_records"
fi

if [ ! -d "${dataset_dir}" ]; then
  mkdir "${dataset_dir}"
fi

# execute the final python script
eval "$script --output_dir=\"${dataset_dir}\""

cp "${out_dir}/config.txt" "${dataset_dir}/config.txt"

echo "cleaning"
rm "${out_dir}/tmp_vocab.txt"
for f in "train" "valid" "test"
do
  rm "${out_dir}/${f}_pfbpe_vocab_${num_merges}.txt"
done
echo "done"
