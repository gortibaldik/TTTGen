from argparse import ArgumentParser
import os

if __name__ == "__main__":
  from utils import resolve_player_name_faults # pylint: disable=import-error
  from constants import BoxScoreEntries # pylint: disable=import-error
else:
  from .utils import resolve_player_name_faults
  from .constants import BoxScoreEntries

def resolve_record(record):
  val, ent, tp, ha = record.split(chr(65512))
  try:
    t = BoxScoreEntries(tp)
    # we know it is a player record, preprocess the entity the same
    # way as we preprocess .jsons
    ent = "_".join(resolve_player_name_faults(" ".join(ent.split("_"))).split())
    if tp == "PLAYER_NAME":
      val = ent
  except:
    # it is a team record, resolve LA -> Los_Angeles
    if tp == "TEAM-CITY" and val == "LA":
      val = "Los_Angeles"
  return val, ent, tp, ha

def main(s, to_be_left, args):
  max_length = 0
  new_lines = []
  with open(os.path.join(args.content_plans_dir, s), 'r', encoding='utf8') as f:
    lines = f.read().strip().split('\n')
    for ix, line in enumerate(lines):
      # jumping over one content plan which is connected to wrong data
      if ix in to_be_left:
        print(line)
        continue
      records = line.strip().split()
      new_records = []
      ixr = 0
      while ixr < len(records):
        record = records[ixr]
        val, ent, tp, ha = resolve_record(record)
        if tp == "FIRST_NAME" and val != "Nene":
          if (ixr + 1 < len(records)):
            next_record = records[ixr + 1]
          else:
            raise RuntimeError(f"not enough records : {ha}, {tp}, {ent}, {val}")

          nval, nent, ntp, nha = resolve_record(next_record)
          if ntp == "SECOND_NAME" and nent == ent and nha == ha:
            val = ent
            tp = "PLAYER_NAME"
          else:
            raise RuntimeError(f"first name not followed by second name : ({ha}, {tp}, {ent}, {val}) ({nha}, {ntp}, {nent}, {nval})")

          ixr += 2
        elif val == "Nene":
          val = ent
          tp = "PLAYER_NAME"
          ixr += 1
        else:
          ixr +=1
        if tp == "SECOND_NAME":
          val = ent
          tp = "PLAYER_NAME"

        new_records.append(f"{chr(65512)}".join([val, ent, tp, ha]))
      if len(new_records) > max_length:
        max_length = len(new_records)
      new_lines.append(new_records)
  with open(os.path.join(args.out_dir, s), 'w', encoding='utf8') as f:
    for line in new_lines:
      print(" ".join(line), file=f)

  return max_length

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("content_plans_dir", type=str)
  parser.add_argument("out_dir", type=str)
  args = parser.parse_args()
  ml = 0
  for s in [("train.txt", [1754, 2685]), ("valid.txt", [183])]:
    nm = main(*s,args)
    if nm > ml: ml = nm

  with open(os.path.join(args.out_dir, "config.txt"), 'w', encoding='utf8') as f:
    print(ml, file=f)
