"""
Data loader and cleaner for KRIXION Hate Speech Detection.

Reads all CSVs under `data/raw/*.csv`, normalizes common source schemas
into a canonical dataframe with columns: `text`, `label`.

References (datasets commonly used in hate/offensive detection):
- Bohra et al. (2018) Hindi Hostility (Bohra dataset)
- Indo-HateSpeech (Indonesian Hate Speech)
- HASOC (Hate Speech and Offensive Content Identification)

Usage (CLI):
	python -m src.data.load_data [--sample N]

This will write the cleaned dataset to `data/clean_data.csv` and print
counts per label to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from src.data.preprocess import preprocess_text


RAW_DIR = Path('data/raw')
OUTPUT_CSV = Path('data/clean_data.csv')

# Common text and label column aliases often seen across datasets
TEXT_CANDIDATES = ['text', 'tweet', 'post', 'content', 'comment', 'sentence']
LABEL_CANDIDATES = ['label', 'class', 'target', 'category', 'y']


def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
	lowered = {c.lower(): c for c in df.columns}
	for name in candidates:
		if name in lowered:
			return lowered[name]
	return None


def _load_single_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	# Try to identify text/label columns
	text_col = _find_first_column(df, TEXT_CANDIDATES)
	label_col = _find_first_column(df, LABEL_CANDIDATES)

	if text_col is None:
		# if no obvious text column, pick first object-like col
		obj_cols = [c for c in df.columns if df[c].dtype == 'object']
		text_col = obj_cols[0] if obj_cols else df.columns[0]

	if label_col is None:
		# If label is missing, create a default neutral label 0
		df['__default_label__'] = 0
		label_col = '__default_label__'

	out = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
	return out


def load_all_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
	raw_dir.mkdir(parents=True, exist_ok=True)
	csv_files: List[Path] = sorted(raw_dir.glob('*.csv'))
	frames: List[pd.DataFrame] = []
	for f in csv_files:
		try:
			frames.append(_load_single_csv(f))
		except Exception as e:
			print(f"Skipping {f}: {e}")
	if not frames:
		# Return empty canonical dataframe if no data yet
		return pd.DataFrame({
			'text': pd.Series(dtype='string'),
			'label': pd.Series(dtype='int64'),
		})
	return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure column types
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError('Input dataframe must have columns: text, label')

    # Apply preprocessing: clean text and detect language
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['text'] = df['text'].astype('string').fillna('')
    
    # Preprocess each text and extract cleaned text + language
    preprocessed = df['text'].apply(lambda t: preprocess_text(t) if t.strip() else ('', 'en'))
    df['text'] = preprocessed.apply(lambda x: x[0])
    df['lang'] = preprocessed.apply(lambda x: x[1])
    
    # Drop empty rows after cleaning
    df = df[df['text'] != '']

    # Normalize label to integers if possible
    # Common mapping: {"hate":1, "offensive":1, "toxic":1, "non-hate":0, "normal":0}
    label_map = {
        'hate': 1, 'offensive': 1, 'toxic': 1, 'abusive': 1, 'hostile': 1,
        'non-hate': 0, 'not-hate': 0, 'normal': 0, 'clean': 0, 'neutral': 0,
    }

    def _normalize_label(v) -> int:
            if pd.isna(v):
                return 0
            # Numbers: preserve 0/1/2 if present, coerce larger values to 2
            if isinstance(v, (int, float)) and not pd.isna(v):
                try:
                    iv = int(v)
                    if iv <= 0:
                        return 0
                    if iv == 1:
                        return 1
                    # any integer >=2 -> Hate (2)
                    return 2
                except Exception:
                    pass

            s = str(v).strip().lower()
            # Common mappings
            if s in {'0', 'none', 'normal', 'clean', 'non-hate', 'not-hate', 'neutral'}:
                return 0
            if s in {'1', 'offensive', 'offence', 'abusive', 'toxic', 'abuse', 'insult'}:
                return 1
            if s in {'2', 'hate', 'hateful', 'hostile', 'severe', 'threat'}:
                return 2
            # Try to pick up common textual tokens
            if any(k in s for k in ['hate', 'kill', 'destroy', 'wipe', 'vermin', 'should be punished']):
                return 2
            if any(k in s for k in ['offensive', 'abuse', 'insult', 'toxic', 'rude']):
                return 1
            # Fallback: treat 'yes','true','y','1' as offensive, otherwise normal
            return 1 if s in {'1', 'true', 'yes', 'y'} else 0

    df['label'] = df['label'].apply(_normalize_label).astype('int64')
    df = df.reset_index(drop=True)
    return df
def save_clean_data(df: pd.DataFrame, path: Path = OUTPUT_CSV) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Load and clean raw CSVs into a canonical dataset.')
    parser.add_argument('--sample', type=int, default=None, help='Optionally write only first N rows')
    args = parser.parse_args()

    df = load_all_raw(RAW_DIR)
    df = clean_dataframe(df)

    if args.sample is not None and args.sample > 0:
        df = df.head(int(args.sample))

    save_clean_data(df, OUTPUT_CSV)

    counts = df['label'].value_counts().sort_index()
    lang_counts = df['lang'].value_counts().sort_index() if 'lang' in df.columns else None
    
    print('Saved to', OUTPUT_CSV)
    print('Total rows:', len(df))
    print('Counts per label:')
    for lbl, cnt in counts.items():
        print(f'  {lbl}: {cnt}')
    
    if lang_counts is not None and not lang_counts.empty:
        print('Counts per language:')
        for lang, cnt in lang_counts.items():
            print(f'  {lang}: {cnt}')
if __name__ == '__main__':
	main()

