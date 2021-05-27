transform_file() {
  file=$1
  echo "Transforming \"$1\""
  sed -i 's/@@ //g' "$1"
  sed -i 's/_/ /g' "$1"
  python3 pp.py "$1"
}

dir=$1
transform_file "${dir}/val_preds.txt"
transform_file "${dir}/val_golds.txt"
transform_file "${dir}/test_preds.txt"
transform_file "${dir}/test_golds.txt"
