summaries_dir=$1
dataset_dir=$2
compilation_dir=$3
out_dir=$4

create_vis() {
  sample=$1
  out_number=$2
  dataset=$3
  python3 visualizations_helper.py "${sample}" "${dataset_dir}" "${summaries_dir}" "${dataset}" "${compilation_dir}/${out_number}/thesis.tex"
  actual_dir=$(pwd)
  cd "${compilation_dir}/${out_number}/"
  make
  cd "${actual_dir}"
  cp "${compilation_dir}/${out_number}/thesis.pdf" "${out_dir}/${out_number}.pdf"
}

create_vis 16  1 "test"
create_vis 247 2 "test"
create_vis 585 3 "test"
create_vis 671 4 "test"
create_vis 132 5 "valid"
create_vis 319 6 "valid"
create_vis 475 7 "valid"
create_vis 697 8 "valid"