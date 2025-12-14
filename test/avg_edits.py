#!/usr/bin/env python3
import csv
import glob
import os
from statistics import mean


def compute_avg_num_edits(csv_path: str) -> float:
    values = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get('num_edits')
            if val is None or val == '':
                continue
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    continue
            # Only keep non-zero edit counts
            if v != 0:
                values.append(v)
    return mean(values) if values else 0.0


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'processed_edits')
    base_dir = os.path.abspath(base_dir)
    csv_files = sorted(glob.glob(os.path.join(base_dir, '*.csv')))

    if not csv_files:
        print('No CSV files found in:', base_dir)
        return

    def detect_lang_from_filename(path: str) -> str:
        name = os.path.basename(path).lower()
        if '_en-' in name or '-en-' in name or 'english' in name:
            return 'en'
        if '_de-' in name or '-de-' in name or 'german' in name:
            return 'de'
        if '_ua-' in name or '-ua-' in name or '_uk-' in name or '-uk-' in name or 'ukrainian' in name:
            return 'ua'
        return ''

    # Accumulate all nonzero num_edits per language across files
    lang_to_values = {'en': [], 'ua': [], 'de': []}

    for path in csv_files:
        lang = detect_lang_from_filename(path)
        if not lang:
            continue
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get('num_edits')
                if val is None or val == '':
                    continue
                try:
                    v = int(val)
                except ValueError:
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                if v != 0:
                    lang_to_values[lang].append(v)

    # Pretty labels without emojis
    order = ['en', 'ua', 'de']
    names = {'en': 'English', 'ua': 'Ukrainian', 'de': 'German'}

    for lang in order:
        vals = lang_to_values.get(lang, [])
        if not vals:
            continue
        avg = mean(vals)
        print(f"* {names[lang]}: {avg:.3f}")


if __name__ == '__main__':
    main()
