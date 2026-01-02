from pathlib import Path
import pandas as pd

RAW_DIR = Path('data/raw')


def inspect():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(RAW_DIR.glob('*.csv'))
    if not csv_files:
        print('No CSV files found in data/raw/')
        return

    for f in csv_files:
        print('\n---')
        print(f'File: {f.name}')
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f'  Failed to read: {e}')
            continue
        print(f'  Rows: {len(df)}')
        print(f'  Columns: {list(df.columns)}')
        print('  Sample:')
        print(df.head(3).to_string(index=False))

        # Try to locate a label/text column
        label_cols = [c for c in df.columns if c.lower() in {'label', 'class', 'target', 'category', 'y'}]
        text_cols = [c for c in df.columns if c.lower() in {'text', 'tweet', 'post', 'content', 'comment', 'sentence'}]
        if label_cols:
            col = label_cols[0]
            print(f'  Detected label column: {col}')
            try:
                vals = df[col].dropna().astype(str).str.strip()
                unique = vals.unique()[:50]
                print(f'  Unique label values (sample): {list(unique)}')
                print('  Counts:')
                print(vals.value_counts().head(20).to_string())
            except Exception as e:
                print(f'   Could not inspect label column: {e}')
        else:
            print('  No label-like column detected')

        if text_cols:
            print(f'  Detected text column(s): {text_cols}')
        else:
            print('  No obvious text column detected')


if __name__ == '__main__':
    inspect()
