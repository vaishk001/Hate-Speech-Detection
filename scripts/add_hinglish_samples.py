import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/dataset_hindi_hinglish.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    new_data = [
        {"text": "tum very bad ho", "label": 1},
        {"text": "tu ek number ka very bad insan hai", "label": 1},
        {"text": "very bad service and very bad person", "label": 1},
        {"text": "tum log bahut bad ho", "label": 1},
        {"text": "you guys are very bad", "label": 1},
        {"text": "bhai tu sach me bahut bad hai", "label": 1},
        {"text": "very bad", "label": 1},
        {"text": "tum bohut bad admi ho", "label": 1},
    ]
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Added {len(new_data)} samples to {csv_path}")
else:
    print(f"{csv_path} does not exist.")
