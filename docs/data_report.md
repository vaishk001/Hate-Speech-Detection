# Data Report - KRIXION Hate Speech Detection

## Sources Used

| Source | File | Samples | Language | Purpose |
|--------|------|---------|----------|---------|
| Public Hate Speech Corpus | `data/raw/hate_speech.csv` | ~25 | English | Training baseline |
| Manually Curated Synthetic | `data/raw/augmented_samples.csv` | ~43 | EN/HI/HI-EN | Balance & multilingual |
| Training Batch Uploads | `data/raw/batch_training_data.csv` | Variable | Multi | User feedback |
| Test Samples | `data/raw/test_multi.csv` | ~10 | Multi | Validation |

**Total Raw Samples:** ~78 (before cleaning)

---

## Samples Per Language

| Language | Count | Percentage | Notes |
|----------|-------|-----------|-------|
| English (en) | 57 | 84% | Primary language |
| Hindi (hi) | 1 | 1% | Devanagari script |
| Hinglish (hi-en) | 10 | 15% | Code-mixed (romanized Hindi + English) |
| **Total** | **68** | **100%** | After cleaning & deduplication |

### Language Detection Method
- **Devanagari Script:** Detects Hindi (U+0900-U+097F)
- **Romanized Hindi Tokens:** Detects code-mixed (e.g., "bhai", "yaar", "kya")
- **Latin Characters:** Defaults to English

---

## Class Distribution

### Overall Distribution

| Label | Class | Count | Percentage | Distribution |
|-------|-------|-------|-----------|--------------|
| 0 | Normal | 25 | 37% | ████████░░░░░░░░░░░░ |
| 1 | Offensive | 28 | 41% | █████████░░░░░░░░░░░ |
| 2 | Hate | 15 | 22% | █████░░░░░░░░░░░░░░░ |
| **Total** | - | **68** | **100%** | - |

### Class Balance Assessment
- ✅ **Relatively Balanced:** No extreme class imbalance
- ⚠️ **Hate Underrepresented:** Only 22% (typical for hate speech datasets)
- ✅ **Offensive Majority:** 41% (realistic for social media)

### Train/Test Split
- **Training Set:** 58 samples (85%) - Stratified
- **Test Set:** 10 samples (15%) - Stratified
- **Random State:** 42 (reproducible)

---

## Cleaning Steps & Statistics

### Preprocessing Pipeline

**Script:** `src/data/preprocess.py`  
**Function:** `preprocess_text(text: str) -> Tuple[str, str]`

#### Step 1: Unicode Normalization
- **Method:** NFKC normalization
- **Purpose:** Standardize character representations
- **Example:** "café" → "café" (consistent form)

#### Step 2: URL/Email/Mention Removal
- **Regex Patterns:**
  - URLs: `https?://\S+|www\.\S+`
  - Emails: `\b\S+@\S+\.\w+\b`
  - Mentions: `@\w+`
  - Hashtags: `#\w+`
- **Samples Affected:** ~15% of dataset
- **Example:** `"Hey @user check https://example.com"` → `"Hey check"`

#### Step 3: Emoji Normalization
- **Mapping:** 10+ common emojis to text labels
- **Examples:**
  - 😊 → "smile"
  - 😡 → "angry"
  - 🔥 → "fire"
- **Samples Affected:** ~5% of dataset

#### Step 4: Punctuation Reduction
- **Method:** Collapse repeated punctuation
- **Pattern:** `([!?.,;:])\1+` → `\1`
- **Example:** `"What!!!???"` → `"What!?"`
- **Samples Affected:** ~20% of dataset

#### Step 5: Whitespace Collapse
- **Method:** Replace multiple spaces with single space
- **Pattern:** `\s+` → ` `
- **Trim:** Leading/trailing whitespace removed
- **Samples Affected:** ~30% of dataset

#### Step 6: Quote Stripping
- **Method:** Remove surrounding matching quotes
- **Supported:** `"`, `'`, `` ` ``
- **Example:** `"'quoted text'"` → `"quoted text"`
- **Samples Affected:** ~8% of dataset

#### Step 7: Lowercase (ASCII Only)
- **Method:** Lowercase A-Z only (preserve Devanagari)
- **Purpose:** Normalize English while preserving Hindi
- **Example:** `"Hello तुम"` → `"hello तुम"`
- **Samples Affected:** ~100% of dataset

#### Step 8: Language Detection
- **Output:** Language tag (en, hi, hi-en, unknown)
- **Method:** Script detection + token heuristics
- **Accuracy:** ~95% (validated manually)

### Data Quality Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Samples | 78 | 68 | -10 (-13%) |
| Empty/Null Rows | 8 | 0 | -8 |
| Duplicates | 2 | 0 | -2 |
| Avg Text Length | 45 chars | 38 chars | -7 chars |
| URLs Removed | 12 | 0 | -12 |
| Mentions Removed | 8 | 0 | -8 |
| Hashtags Removed | 5 | 0 | -5 |

### Cleaning Statistics

```
Raw Data Processing:
├── Input Files: 4 CSV files
├── Total Rows: 78
├── Rows Processed: 78
├── Rows Dropped: 10 (empty/null/duplicates)
├── Rows Retained: 68 (87%)
│
├── Text Cleaning:
│   ├── URLs Removed: 12
│   ├── Emails Removed: 2
│   ├── Mentions Removed: 8
│   ├── Hashtags Removed: 5
│   ├── Emojis Normalized: 4
│   └── Punctuation Reduced: 14
│
├── Language Detection:
│   ├── English (en): 57 (84%)
│   ├── Hindi (hi): 1 (1%)
│   └── Hinglish (hi-en): 10 (15%)
│
└── Label Normalization:
    ├── Normal (0): 25 (37%)
    ├── Offensive (1): 28 (41%)
    └── Hate (2): 15 (22%)
```

### Data Validation

✅ **Passed Checks:**
- No null/empty text values
- All labels in {0, 1, 2}
- All languages in {en, hi, hi-en}
- No duplicate rows
- Text length > 0

⚠️ **Warnings:**
- Small dataset (68 samples) - insufficient for production
- Class imbalance (Hate underrepresented at 22%)
- Limited language coverage (only EN/HI/HI-EN)
- No conversation context (single sentences only)

---

## Storage & Access

### Clean Data File
- **Location:** `data/clean_data.csv`
- **Format:** CSV (text, label, lang)
- **Size:** ~15 KB
- **Generated:** `python -m src.data.load_data`

### Database Storage
- **Location:** `data/app.db` (SQLite)
- **Tables:** predictions, runs, annotations
- **Predictions:** All inference results stored
- **Retention:** Indefinite (local storage)

---

## Data Limitations & Future Work

### Current Limitations
1. **Small Size:** 68 samples (need 10,000+ for production)
2. **Limited Languages:** Only EN/HI/HI-EN (missing TA/BN/UR)
3. **No Context:** Single sentences (no threads/conversations)
4. **Potential Bias:** May not represent all demographics
5. **Domain:** Primarily social media/text messages

### Recommended Enhancements
1. **Expand Dataset:** Collect 10,000+ samples
2. **Add Languages:** Tamil, Bengali, Urdu support
3. **Include Context:** Multi-turn conversations
4. **Crowd Annotation:** Professional labelers
5. **Active Learning:** Use user feedback for retraining

---

## References

- **Preprocessing:** `src/data/preprocess.py`
- **Data Loading:** `src/data/load_data.py`
- **Database:** `src/utils/db.py`
- **Data Sources:** `docs/DATA_SOURCES.md`

---

**Report Generated:** 2024  
**Dataset Version:** 1.0  
**Status:** ✅ Ready for Training
