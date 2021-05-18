try:
    from utils import OccurrenceDict, create_tp_vocab, create_ha_vocab # pylint: disable=import-error
    from extract_matches import extract_matches_from_json # pylint: disable=import-error
except:
    from .utils import OccurrenceDict, create_tp_vocab, create_ha_vocab
    from .extract_matches import extract_matches_from_json

import os
import numpy as np
import tensorflow as tf

_create_dataset_descr = "create_dataset"

def create_dataset_parser(subparsers):
    create_dataset_parser = subparsers.add_parser(_create_dataset_descr)
    create_dataset_parser.add_argument(
        "--preproc_summaries_dir",
        type=str,
        help="path to directory with output of create_dataset.sh",
        required=True
    )
    create_dataset_parser.add_argument(
        '--output_dir',
        type=str,
        help="directory where the outputs will be saved",
        required=True
    )
    create_dataset_parser.add_argument(
        '--order_records',
        action='store_true',
        help="If true, the input tables will contain teams in the first records and players" +\
                "sorted by their point totals"
    )
    create_dataset_parser.add_argument(
        '--content_plans_dir',
        help="path to directory with content plans created by Puduppully et al. 2019",
        type=str,
        default=None
    )
    create_dataset_parser.add_argument(
        "--to_npy",
        help="save the output to .npy format (indices in npy arrays)",
        action="store_true"
    )
    create_dataset_parser.add_argument(
        "--to_txt",
        help="save the output to .txt format (indices in txt)",
        action="store_true"
    )
    create_dataset_parser.add_argument(
        "--to_tfrecord",
        help="save the output to .tfrecord format",
        action="store_true"
    )

def create_prepare(args, set_names, input_paths):
    suffix = ""
    if not args.to_npy and not args.to_txt and not args.to_tfrecord:
        raise RuntimeError("Exactly one from --to_npy --to_txt --to_tfrecord should be specified")
    elif (args.to_npy and args.to_txt) or (args.to_npy and args.to_tfrecord) or (args.to_txt and args.to_tfrecord):
        raise RuntimeError("Multiple mutually exclusive options specified (--to_npy and --to_txt)")
    elif args.to_npy:
        suffix = ".npy"
    elif args.to_txt:
        suffix = ".txt"
    elif args.to_tfrecord:
        suffix = ".tfrecord"

    # prepare input paths
    for ix, pth in enumerate(set_names):
        if (pth == "test") or (args.content_plans_dir == None):
            cplan = None
        else:
            cplan = os.path.join(args.content_plans_dir, pth + ".txt")
        input_paths[ix] = (os.path.join(args.preproc_summaries_dir, pth + "_prepared.txt"), input_paths[ix], cplan)

    # prepare output paths
    output_paths = []
    for ix, pth in enumerate(set_names):
        pth = os.path.join(args.output_dir, pth)
        if suffix != ".tfrecord":
            output_paths.append((pth + "_in" + suffix, pth + "_target" + suffix, pth + "_cp" + suffix))
        else:
            output_paths.append([pth + suffix])

    # prepare vocab of unique tokens from summaries
    token_vocab_path = os.path.join(args.preproc_summaries_dir, "token_vocab.txt")
    tk_vocab = OccurrenceDict.load(token_vocab_path, basic_dict=True)

    # prepare vocab of cell values
    cell_vocab_path = os.path.join(args.preproc_summaries_dir, "cell_vocab.txt")
    cl_vocab = OccurrenceDict.load(cell_vocab_path)

    # join the vocabs
    tk_vocab.update(cl_vocab)
    tk_vocab = tk_vocab.sort()
    tk_vocab.save(os.path.join(args.output_dir, "all_vocab.txt"))

    # get max table length and max summary length
    mlt, mls = 0, 0
    with open(os.path.join(args.preproc_summaries_dir, "config.txt"), 'r') as f:
        tokens = [int(n) for n in f.read().strip().split('\n')]
        mlt, mls = tokens[0], tokens[1]

    # get max content plan length
    mlcp = 0
    if args.content_plans_dir is not None:
        with open(os.path.join(args.content_plans_dir, "config.txt"), 'r') as f:
            mlcp = int(f.read().strip().split('\n')[0])
    return input_paths, output_paths, { "tk_vocab": tk_vocab
                                      , "mlt": mlt
                                      , "mls": mls
                                      , "mlcp": mlcp
                                      , "order_records": args.order_records }

def create_content_plan_ids( content_plan_path
                           , mlcp
                           , pad_value
                           , tables
                           , logger):
    plan_ids = np.full(shape=(len(tables), mlcp), fill_value=pad_value, dtype=np.int16)
    with open(content_plan_path, 'r', encoding='utf8') as f:
        logger(f"Working with {content_plan_path}")
        lines = f.read().strip().split('\n')
        if len(lines) != len(tables):
            raise RuntimeError(f"len(lines) : {len(lines)}; len(tables) : {len(tables)}")
        for ixt, (line, table) in enumerate(zip(lines, tables)):
            records = line.strip().split()
            for ixr, record in enumerate(records):
                val, ent, tp, ha = record.split(chr(65512))
                resolved_id = -1
                for ixx, r in enumerate(table):
                    if (r.ha == ha) and (r.type == tp) and \
                       ("_".join(r.entity.split()) == ent) and (r.value == val):
                        resolved_id = ixx + 1 # pad
                        break
                if resolved_id == -1:
                    for rr in table:
                        if rr.type == "PLAYER_NAME":
                            print(f"{rr.ha}, {rr.entity}, {rr.value}")
                    raise RuntimeError(f"line {ixt}, file {content_plan_path}, record: {ha}, {tp}, {ent}, {val}")
                plan_ids[ixt, ixr] = resolved_id
    return plan_ids


def save_np_to_txt( _np_in: np.ndarray
                  , _np_target: np.ndarray
                  , cplan_ids: np.ndarray
                  , _out_paths
                  , _logger
                  , summary_path
                  , json_path):
    _out_in_path = _out_paths[0]
    _out_target_path = _out_paths[1]
    _out_cp_path = _out_paths[2]
    if cplan_ids is not None:
        _logger(f"content plans -> {_out_cp_path}")
    _logger(f"summaries {summary_path} -> {_out_target_path}")
    _logger(f"tables {json_path} -> {_out_in_path}")
    _logger("--- saving to .txt")
    with open(_out_in_path, 'w') as f:
        for _m_ix in range(_np_in.shape[0]):
            for _t_ix in range(_np_in.shape[2]):
                for _v_ix in range(_np_in.shape[1]):
                    print(_np_in[_m_ix, _v_ix, _t_ix], end="|", file=f)
                print(end=" ", file=f)
            print(file=f)

    with open(_out_target_path, 'w') as f:
        for _m_ix in range(_np_target.shape[0]):
            for _s_ix in range(_np_target.shape[1]):
                print(_np_target[_m_ix, _s_ix], end=" ", file=f)
            print(file=f)
    
    if cplan_ids is not None:
        with open(_out_cp_path, 'w') as f:
            for _m_ix in range(cplan_ids.shape[0]):
                for _s_ix in range(cplan_ids.shape[1]):
                    print(cplan_ids[_m_ix, _s_ix], end=" ", file=f)
                print(file=f)

def save_np_to_tfrecord( _np_in: np.ndarray
                       , _np_target: np.ndarray
                       , cplan_ids: np.ndarray
                       , _out_path
                       , _logger
                       , summary_path
                       , json_path):
    if cplan_ids is not None:
        _logger(f"content plans -> {_out_path}")
    _logger(f"summaries {summary_path} -> {_out_path}")
    _logger(f"tables {json_path} ->> {_out_path}")
    if cplan_ids is None:
        data = tf.data.Dataset.from_tensor_slices(
            ( _np_target
            , _np_in[:, 0, :]
            , _np_in[:, 1, :]
            , _np_in[:, 2, :]
            , _np_in[:, 3, :])
        )
    else:
        data = tf.data.Dataset.from_tensor_slices(
            ( _np_target
            , cplan_ids
            , _np_in[:, 0, :]
            , _np_in[:, 1, :]
            , _np_in[:, 2, :]
            , _np_in[:, 3, :])          
        )

    # code from https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
    def write_map_fn(_summary, types, entities, values, has):
        return tf.io.serialize_tensor(tf.concat([_summary, types, entities, values, has], -1))
    
    def write_map_fn_cp(_summary, _cp, types, entities, values, has):
        return tf.io.serialize_tensor(tf.concat([_summary, _cp, types, entities, values, has], -1))
    
    if cplan_ids is None:
        data = data.map(write_map_fn)
    else:
        data = data.map(write_map_fn_cp)
    writer = tf.data.experimental.TFRecordWriter(_out_path)
    writer.write(data)


def assign_ix_or_unk(dct, key, unk_stat):
    if key in dct:
        return dct[key]
    else:
        unk_stat.increment_unk_stat()
        return dct[unk_stat.get_unk()]


class UnkStat:
    def __init__(self, tk_vocab):
        self._unk_stat = 0
        self._unk_token = tk_vocab.get_unk()

    def increment_unk_stat(self):
        self._unk_stat += 1
    
    def get_unk_stat(self):
        return self._unk_stat
    
    def get_unk(self):
        return self._unk_token


def create_dataset( input_paths
                  , output_paths
                  , tk_vocab
                  , mlcp # max length content plan
                  , mls # max length summary
                  , mlt # max length table
                  , order_records
                  , logger):
    summary_path, json_path, cplan_path = input_paths

    tables = [ m.records for m in extract_matches_from_json( json_path
                                                           , word_dict=None
                                                           , process_summary=False
                                                           , order_records=order_records)]

    with open(summary_path, 'r') as f:
        file_content = f.read().strip().split('\n')


    summaries = []
    for line in file_content:
        summaries.append(line)

    tk_to_ix = tk_vocab.to_dict()

    tp_vocab = create_tp_vocab()
    tp_to_ix = tp_vocab.to_dict()

    ha_vocab = create_ha_vocab()
    ha_to_ix = ha_vocab.to_dict()

    pad_value = tk_to_ix[tk_vocab.get_pad()]
    if pad_value != tp_to_ix[tp_vocab.get_pad()] or pad_value != ha_to_ix[ha_vocab.get_pad()]:
        raise RuntimeError("Different padding values in type and token vocabs!")
    
    cplan_ids = None
    if cplan_path is not None:
        cplan_ids = create_content_plan_ids( cplan_path
                                           , mlcp
                                           , pad_value
                                           , tables
                                           , logger)

    np_in = np.full(shape=[len(tables), 4, mlt], fill_value=pad_value, dtype=np.int16)
    # add space for special tokens
    np_target = np.full(shape=[len(tables), mls + 2], fill_value=pad_value, dtype=np.int16)
    unk_stat = UnkStat(tk_vocab)

    for m_ix, (table, summary) in enumerate(zip(tables, summaries)):
        for t_ix, record in enumerate(table):
            np_in[m_ix, 0, t_ix] = tp_to_ix[record.type] 
            np_in[m_ix, 1, t_ix] = assign_ix_or_unk( tk_to_ix
                                                   , "_".join(record.entity.strip().split())
                                                   , unk_stat)
            np_in[m_ix, 2, t_ix] = assign_ix_or_unk( tk_to_ix
                                                   , record.value
                                                   , unk_stat)
            np_in[m_ix, 3, t_ix] = ha_to_ix[record.ha]
        np_target[m_ix, 0] = tk_to_ix[tp_vocab.get_bos()]
        summary_tokens = summary.strip().split()
        for s_ix, subword in enumerate(summary_tokens):
            np_target[m_ix, s_ix+1] = assign_ix_or_unk( tk_to_ix
                                                      , subword
                                                      , unk_stat)
        np_target[m_ix, len(summary_tokens) + 1]  = tk_to_ix[tp_vocab.get_eos()]
    
    logger(f"{output_paths[0]} : {unk_stat.get_unk_stat()} tokens assigned for OOV words")

    extension = os.path.splitext(output_paths[0])[1]
    if extension == ".txt":
        save_np_to_txt( np_in
                      , np_target
                      , cplan_ids
                      , output_paths
                      , logger
                      , summary_path
                      , json_path)
    elif extension == ".npy":
        logger(f"summaries {summary_path} -> {output_paths[1]}")
        logger(f"tables {json_path} -> {output_paths[0]}")
        logger("--- saving to .npy")
        np.save(output_paths[0], np_in)
        np.save(output_paths[1], np_target)
        if cplan_ids is not None:
            np.save(output_paths[2], cplan_ids)
    elif extension == ".tfrecord":
        save_np_to_tfrecord( np_in
                           , np_target
                           , cplan_ids
                           , output_paths[0]
                           , logger
                           , summary_path
                           , json_path)