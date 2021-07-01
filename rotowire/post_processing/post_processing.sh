transform_file() {
  file=$1
  echo "Transforming \"$1\""
  sed -i 's/@@ //g' "$1"
  sed -i 's/_/ /g' "$1"
  python3 helper_postprocessing.py "$1"
}

file=$1
transform_file "${file}"
