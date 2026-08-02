import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Split test.csv into per-row .txt (input) and .json (gold entities) files for evaluation"
    )
    parser.add_argument("--csv", default="data/medor/test.csv")
    parser.add_argument("--out-dir", default="data/medor/eval")
    args = parser.parse_args()

    txt_dir = os.path.join(args.out_dir, "txt")
    gold_dir = os.path.join(args.out_dir, "gold")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    for i, row in df.iterrows():
        with open(os.path.join(txt_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
            f.write(row["input_text"])
        entities = json.loads(row["response"])
        with open(os.path.join(gold_dir, f"{i}.json"), "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote {len(df)} samples to {txt_dir} and {gold_dir}")


if __name__ == "__main__":
    main()
