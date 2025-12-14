import pandas as pd
from grampy.text import AnnotatedText
import difflib
import argparse
import os


def show_diff(a, b):
    # Shows word-by-word difference
    diff = difflib.ndiff(a.split(), b.split())
    print('Difference:')
    print('\n'.join(diff))


def main():
    parser = argparse.ArgumentParser(description='Check alignment accuracy for GEC datasets')
    parser.add_argument('--lang', default='en', choices=['en', 'ua', 'de', 'es'], 
                       help='Language to check (default: en)')
    parser.add_argument('--file', help='Specific CSV file to check (overrides --lang)')
    args = parser.parse_args()
    
    # Load the appropriate CSV file based on language or specific file
    if args.file:
        csv_file = args.file
        lang_label = os.path.basename(csv_file).split('-')[0].upper()
    else:
        csv_file = f'data/processed/{args.lang}-judge.csv'
        # For test files, use a different naming pattern
        if args.lang == 'es' and not os.path.exists(csv_file):
            csv_file = f'data/processed/{args.lang}-test.csv'
        lang_label = args.lang.upper()
    
    print(f"Checking alignment accuracy for {lang_label} dataset: {csv_file}")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found. Please run processing/{args.lang}.py first.")
        return
    
    num_errors = 0
    
    for idx, row in df.iterrows():
        aligned = row['aligned']
        tgt = row['tgt']
        corrected = AnnotatedText(aligned).get_corrected_text()

        if corrected != tgt:
            print(f"Row {idx}:")
            print(f"  src:         {row['src']}")
            print(f"  tgt:         {tgt}")
            print(f"  aligned:     {aligned}")
            print(f"  corrected:   {corrected}")
            #show_diff(corrected, tgt)
            print("-" * 30)
            num_errors += 1

    print(f"\nAlignment Accuracy Results for {lang_label}:")
    print(f"Total samples: {len(df)}")
    print(f"Alignment errors: {num_errors}")
    print(f"Error rate: {num_errors / len(df) * 100:.1f}%")
    print(f"Accuracy: {(len(df) - num_errors) / len(df) * 100:.1f}%")


if __name__ == "__main__":
    main()