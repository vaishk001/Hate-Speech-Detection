import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/dataset_hindi_hinglish.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    new_data = [
        # Sarcasm (Normal looking words but hate/offensive intent)
        {"text": "waah bhai kya baat hai, tum to bade hoshiyar ho", "label": 1},
        {"text": "bahut badhiya, aur kitna giroge tum", "label": 1},
        {"text": "kya kehney aapke, ek number ke jhoothe ho", "label": 1},
        {"text": "waah bhai waah, pura desh bech diya", "label": 2},
        {"text": "bahut achha kaam kiya, sabko marwa diya", "label": 2},
        {"text": "bade aaye gyan dene wale, khud ka dimaag kharab hai", "label": 1},
        
        # Contextual group targeting
        {"text": "in logo ko to desh se bahar nikal dena chahiye", "label": 2},
        {"text": "woh log kabhi nahi sudhrenge, sabko khatam kar do", "label": 2},
        {"text": "their kind is always making trouble, ban them all", "label": 2},
        {"text": "aise logo ko society me jeene ka haq nahi hai", "label": 2},
        
        # English sarcasm
        {"text": "oh great, another genius is here to ruin everything", "label": 1},
        {"text": "what a brilliant idea, let's just destroy the whole thing", "label": 1},
        {"text": "wow, you are so smart that you broke it", "label": 1},
        
        # Mild but contextual
        {"text": "you people are the real problem", "label": 1},
        {"text": "sab tumhare jaise logo ki wajah se ho raha hai", "label": 1},
        
        # Hate
        {"text": "khatam kar do in sab ko", "label": 2},
        {"text": "mar dalenge aise logo ko", "label": 2},
    ]
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Added {len(new_data)} samples to {csv_path}")
else:
    print(f"{csv_path} does not exist.")
