# Code copied from https://github.com/tyliupku/wiki2bio
# purpose : preprocessing WIKIBIO dataset
# original authors : Tianyu Liu, Kexiang Wang, Lei Sha, Baobao Chang, Zhifang Sui
# the original script is changed to such a way, that :
#   - data is saved to .npy archives instead of plain txt
#   - data is filtered and padded to FILTER_TABLE_SIZE and FILTER_SUMMARY_SIZE respectively

import os
import re
import time
import numpy as np
from vocab import Vocab

FILTER_TABLE_SIZE = 100
FILTER_SUMMARY_SIZE = 75
deleted_indices = {}


def save_filtered(number_data, output_files, filter_size, deleted_indices = None):
    for id, (data, name) in enumerate(number_data):
        print(f"{output_files[id]} <> {name}")
        data_array = np.zeros([len(data), filter_size], dtype=np.short)
        cnt = -1
        for row, line_items in enumerate(data):
            # filtering
            if (deleted_indices is not None) and (not deleted_indices[name][row]):
                continue
            cnt += 1
            for col, item in enumerate(line_items):
                data_array[cnt, col] = int(item)
        with open(output_files[id], "wb") as f:
            np.save(f, data_array[:cnt + 1])


def split_infobox():
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the beginning of a field
    """
    bwfile = ["processed_data/train/train.box.val",
              "processed_data/valid/valid.box.val",
              "processed_data/test/test.box.val"]
    bffile = ["processed_data/train/train.box.lab",
              "processed_data/valid/valid.box.lab",
              "processed_data/test/test.box.lab"]
    bpfile = ["processed_data/train/train.box.pos",
              "processed_data/valid/valid.box.pos",
              "processed_data/test/test.box.pos"]
    boxes = ["original_data/train.box", "original_data/valid.box", "original_data/test.box"]

    all_sets_values, all_sets_labels, all_sets_positions = [], [], []
    for fboxes in boxes:
        # read all the contents and split the file to lines
        set_content = open(fboxes, "r").read().strip().split('\n')
        set_values, set_labels, set_positions = [], [], []
        for line in set_content:
            # on one line, there is information about exactly one box
            line_items = line.split('\t')
            line_values, line_labels, line_positions = [], [], []
            for label_value in line_items:
                # format of single label_value: label_N:value
                if len(label_value.split(':')) > 2:
                    continue

                original_label, original_value = label_value.split(':')
                # in the dataset when a field is not readable (e.g. image fields)
                # the value is "<none>"
                if '<none>' in original_value:
                    continue
                # check for other anomalies, which shouldn't be present
                # in the dataset
                if original_value.strip() == '' or original_label.strip() == '':
                    print(f"field: {original_label}, label: {original_value}")
                    raise RuntimeError("Unable to continue!")

                # labels are in format: label_N
                # extract label part
                label = re.sub("_[1-9]\d*$", "", original_label)
                if label.strip() == "":
                    continue
                line_values.append(original_value)
                line_labels.append(label)
                if re.search("_[1-9]\d*$", original_label):
                    # don't allow any higher positions than 30
                    # if there are some, keep it as 30
                    position = int(original_label.split('_')[-1])
                    line_positions.append(position if position <= 30 else 30)
                else:
                    line_positions.append(1)
            set_values.append(line_values)
            set_labels.append(line_labels)
            set_positions.append(line_positions)
        all_sets_values.append(set_values)
        all_sets_labels.append(set_labels)
        all_sets_positions.append(set_positions)

    def write_string_set( string_sets
                        , train_test_val_output_file
                        , filter_size : int
                        , memory_deleted_indices : bool = False):
        for id, (set, name) in enumerate(zip(string_sets, ["train", "valid", "test"])):
            with open(train_test_val_output_file[id], "w+") as f:
                if memory_deleted_indices:
                    print(f"saving deleted indices of {name} set for file {train_test_val_output_file[id]}")
                for id, line_items in enumerate(set):
                    # ignore tables, where we already ignore the summaries
                    if not deleted_indices[name][id]:
                        continue
                    # filter too large tables
                    if len(line_items) > filter_size:
                        if memory_deleted_indices :
                            deleted_indices[name][id] = False
                        continue
                    for item in line_items:
                        f.write(str(item) + " ")
                    f.write('\n')

    write_string_set(all_sets_values, bwfile, FILTER_TABLE_SIZE, memory_deleted_indices=True)
    write_string_set(all_sets_labels, bffile, FILTER_TABLE_SIZE)
    save_filtered(zip(all_sets_positions, ["train", "valid", "test"]), bpfile, FILTER_TABLE_SIZE, deleted_indices)


def reverse_pos():
    # get the position counted from the end of a field
    bpfile = ["processed_data/train/train.box.pos", "processed_data/valid/valid.box.pos", "processed_data/test/test.box.pos"]
    bwfile = ["processed_data/train/train.box.rpos", "processed_data/valid/valid.box.rpos", "processed_data/test/test.box.rpos"]

    for id, (file_name, set_type) in enumerate(zip(bpfile, ["train", "valid", "test"])):
        data_array = np.load(file_name)
        reverse_pos = []
        for row in data_array:
            tmp_pos = []
            single_pos = []
            for position in row:
                if position == 0.0:
                    continue
                if position == 1.0 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(str(position))
            single_pos.extend(tmp_pos[::-1])
            reverse_pos.append(single_pos)
        save_filtered([(reverse_pos, set_type)], [bwfile[id]], FILTER_TABLE_SIZE)


def check_generated_box():
    ftrain = ["processed_data/train/train.box.val",
              "processed_data/train/train.box.lab",
              "processed_data/train/train.box.pos",
              "processed_data/train/train.box.rpos"]
    ftest  = ["processed_data/test/test.box.val",
              "processed_data/test/test.box.lab",
              "processed_data/test/test.box.pos",
              "processed_data/test/test.box.rpos"]
    fvalid = ["processed_data/valid/valid.box.val",
              "processed_data/valid/valid.box.lab",
              "processed_data/valid/valid.box.pos",
              "processed_data/valid/valid.box.rpos"]
    for case in [ftrain, ftest, fvalid]:
        vals = open(case[0], 'r').read().strip().split('\n')
        labs = open(case[1], 'r').read().strip().split('\n')
        poses = np.load(case[2])
        rposes = np.load(case[3])
        assert len(vals) == len(labs)
        assert len(poses) == len(labs)
        assert len(rposes) == len(poses)
        for val, lab, pos, rpos in zip(vals, labs, poses, rposes):
            vval = val.strip().split(' ')
            llab = lab.strip().split(' ')
            ppos = pos
            rrpos = rpos
            if len(vval) != len(llab) or FILTER_TABLE_SIZE != len(ppos) or len(ppos) != len(rrpos):
                print(case)
                print(val)
                print(len(vval))
                print(len(llab))
                print(len(ppos))
                print(len(rrpos))
            assert len(vval) == len(llab)
            assert FILTER_TABLE_SIZE == len(ppos)
            assert len(ppos) == len(rrpos)


def split_summary_for_rouge():
    bpfile = ["original_data/test.summary", "original_data/valid.summary"]
    bwfile = ["processed_data/test/test_split_for_rouge/", "processed_data/valid/valid_split_for_rouge/"]
    for i, fi in enumerate(bpfile):
        fread = open(fi, 'r')
        k = 0
        for line in fread:
            with open(bwfile[i] + 'gold_summary_' + str(k), 'w') as sw:
                sw.write(line.strip() + '\n')
            k += 1
        fread.close()


def table2id():
    fvals = ['processed_data/train/train.box.val',
             'processed_data/test/test.box.val',
             'processed_data/valid/valid.box.val']
    flabs = ['processed_data/train/train.box.lab',
             'processed_data/test/test.box.lab',
             'processed_data/valid/valid.box.lab']
    fsums = ['original_data/train.summary',
             'original_data/test.summary',
             'original_data/valid.summary']
    train_test_valid = ["train", "test", "valid"]
    fvals2id = ['processed_data/train/train.box.val.id',
                'processed_data/test/test.box.val.id',
                'processed_data/valid/valid.box.val.id']
    flabs2id = ['processed_data/train/train.box.lab.id',
                'processed_data/test/test.box.lab.id',
                'processed_data/valid/valid.box.lab.id']
    fsums2id = ['processed_data/train/train.summary.id',
                'processed_data/test/test.summary.id',
                'processed_data/valid/valid.summary.id']
    vocab = Vocab()

    def str_to_ids(input_files, output_files, vocabf, filter_size : int, filter : bool = False):
        for id, (input_file, name) in enumerate(input_files):
            with open(input_file, 'r') as f:
                input_content = f.read().strip().split('\n')

            # fill data_array with indices of items from the input_file
            data_array = np.zeros([len(input_content), filter_size], dtype=np.short)
            cnt = -1
            for row, line in enumerate(input_content):
                items = line.strip().split(' ')
                if filter and (not deleted_indices[name][row]):
                    continue
                if len(items) > filter_size:
                    raise RuntimeError(f"Invalid input file : {input_file}")
                cnt += 1
                for column, item in enumerate(items):
                    data_array[cnt, column] = vocabf(item)

            data_array = data_array[:cnt+1]

            # save the array with indices
            with open(output_files[id], 'wb') as f:
                np.save(f, data_array)

    str_to_ids(zip(fvals, train_test_valid), fvals2id, vocab.word2id, FILTER_TABLE_SIZE)
    str_to_ids(zip(flabs, train_test_valid), flabs2id, vocab.key2id, FILTER_TABLE_SIZE)
    str_to_ids(zip(fsums, train_test_valid), fsums2id, vocab.word2id, FILTER_SUMMARY_SIZE, filter=True)
    # for k, (ff, name) in enumerate(zip(fsums, ["train", "test", "valid"])):
    #     print(f"{ff} <> {name}")
    #     fi = open(ff, 'r')
    #     fo = open(fsums2id[k], 'w')
    #     for id, line in enumerate(fi):
    #         if not deleted_indices[name][id]:
    #             continue
    #         items = line.strip().split()
    #         fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
    #     fi.close()
    #     fo.close()


def traverse_summaries():
    """
    Traverse all the summaries and add all the lines
    which are too long to the array of deleted indices
    """
    fsums = ['original_data/train.summary',
             'original_data/test.summary',
             'original_data/valid.summary']
    for ff, name in zip(fsums, ["train", "test", "valid"]):
        deleted_indices[name] = []
        print(f"{ff} <> {name}")
        with open(ff, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) > FILTER_SUMMARY_SIZE:
                    deleted_indices[name].append(False)
                else:
                    deleted_indices[name].append(True)


def preprocess():
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field.
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("traversing the summaries to fill up deleted_indices")
    traverse_summaries()
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()
    split_infobox()
    reverse_pos()
    duration = time.time() - time_start
    print("extract finished in %.3f seconds" % float(duration))

    print("spliting test and valid summaries for ROUGE evaluation ...")
    time_start = time.time()
    split_summary_for_rouge()
    duration = time.time() - time_start
    print("split finished in %.3f seconds" % float(duration))

    print("turning words and field types to ids ...")
    time_start = time.time()
    table2id()
    duration = time.time() - time_start
    print("idlization finished in %.3f seconds" % float(duration))


def make_dirs():
    os.mkdir("results/")
    os.mkdir("results/res/")
    os.mkdir("results/evaluation/")
    os.mkdir("processed_data/")
    os.mkdir("processed_data/train/")
    os.mkdir("processed_data/test/")
    os.mkdir("processed_data/valid/")
    os.mkdir("processed_data/test/test_split_for_rouge/")
    os.mkdir("processed_data/valid/valid_split_for_rouge/")


if __name__ == '__main__':
    make_dirs()
    preprocess()
    check_generated_box()
    print("check done")