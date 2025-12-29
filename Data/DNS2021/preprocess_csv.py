import os
import time
import zipfile
import shutil
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("=== CIC-Bell-DNS-2021 CSV Preprocessing ===")
start_time = time.time()

# =========================
# CONFIG
# =========================
DATASET_URL = "http://cicresearch.ca/CICDataset/CICBellDNS2021/Dataset/"
ZIP_NAME = "CSVs-20240207T040926Z-001.zip"
ZIP_URL = DATASET_URL + ZIP_NAME

BASE_DIR = "Data/DNS2021/csv"
ZIP_PATH = os.path.join(BASE_DIR, ZIP_NAME)

RANDOM_STATE = 42
TEST_SIZE = 0.30      # 70 / 15 / 15
VAL_RATIO = 0.50

# =========================
# UTILS
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def download_zip():
    ensure_dir(BASE_DIR)

    if os.path.exists(ZIP_PATH):
        print("ZIP already exists, skipping download.")
        return

    print(f"Downloading {ZIP_NAME} ...")
    r = requests.get(ZIP_URL, stream=True, timeout=120)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100
            print(f"\rProgress: {pct:.1f}%", end="", flush=True)

    print("\nDownload complete ✓")

def extract_csv():
    print("Extracting CSV files...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = [m for m in z.namelist() if m.endswith(".csv")]
        z.extractall(BASE_DIR)

    # Flatten structure
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                dst = os.path.join(BASE_DIR, f)
                if src != dst and not os.path.exists(dst):
                    shutil.move(src, dst)

    print("Extraction complete ✓")

# =========================
# DOWNLOAD & EXTRACT
# =========================
download_zip()
extract_csv()

# =========================
# LOAD CSVs + ASSIGN LABEL
# =========================
dfs = []

for file in os.listdir(BASE_DIR):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(BASE_DIR, file)

    try:
        try:
            df_temp = pd.read_csv(file_path, engine="c", low_memory=False)
        except Exception:
            df_temp = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"Failed to load {file}: {e}")
        continue

    fname = file.lower()
    if "benign" in fname:
        label = "benign"
    elif "spam" in fname:
        label = "spam"
    elif "phishing" in fname:
        label = "phishing"
    elif "malware" in fname:
        label = "malware"
    else:
        raise RuntimeError(f"Cannot infer label from filename: {file}")

    df_temp["Label"] = label
    dfs.append(df_temp)

    print(f"Loaded {file:20s} | rows={len(df_temp):6d} | label={label}")

df = pd.concat(dfs, ignore_index=True)
print("\nCombined dataset shape:", df.shape)
print("\nLabel distribution:\n", df["Label"].value_counts())

# =========================
# CLEAN NaN / INF
# =========================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# =========================
# FEATURES / LABEL - Select exactly 32 features
# =========================
y = df["Label"]

# Map the 32 features from documentation to CSV columns
# Note: Some features don't exist in CSV, so we use closest matches
feature_mapping = {
    # Lexical Features (F1-F14)
    "F1": "subdomain",           # F1: Subdomain
    "F2": "tld",                 # F2: TLD
    "F3": "sld",                 # F3: SLD (keep as feature)
    "F4": "len",                 # F4: Len
    "F5": "numeric_percentage",  # F5: Numeric percentage
    "F6": "char_distribution",   # F6: Character distribution
    "F7": "entropy",             # F7: Entropy
    "F8": "1gram",               # F8: 1-gram
    "F9": "2gram",               # F9: 2-gram
    "F10": "3gram",              # F10: 3-gram
    "F11": "longest_word",       # F11: Longest word
    "F12": "shortened",          # F12: Distance from bad words (NOT IN CSV, use shortened as proxy)
    "F13": "typos",              # F13: Typos
    "F14": "obfuscate_at_sign",  # F14: Obfuscation
    
    # DNS Statistical (F15-F21) - using available columns
    "F15": "Country",            # F15: Unique country → use Country
    "F16": "ASN",                # F16: Unique ASN → use ASN
    "F17": "TTL",                # F17: Unique TTL → use TTL
    "F18": "hex_32",             # F18: Unique IP (NOT IN CSV) → use hex_32
    "F19": "dec_32",             # F19: Unique domain (NOT IN CSV) → use dec_32
    "F20": "hex_8",              # F20: TTL mean (NOT IN CSV) → use hex_8
    "F21": "dec_8",              # F21: TTL variance (NOT IN CSV) → use dec_8
    
    # Third Party (F22-F32)
    "F22": "puny_coded",         # F22: Domain name (identifier, skip) → use puny_coded
    "F23": "Registrar",          # F23: Registrar
    "F24": "oc_8",               # F24: Registrant name (identifier, skip) → use oc_8
    "F25": "Creation_Date_Time", # F25: Creation date time
    "F26": "Emails",             # F26: Emails
    "F27": "Domain_Age",         # F27: Domain age
    "F28": "Organization",      # F28: Organization
    "F29": "State",              # F29: State
    "F30": "Country",            # F30: Country (duplicate of F15, but keeping for 32 features)
    "F31": "Name_Server_Count",  # F31: Name server count
    "F32": "Alexa_Rank",         # F32: Alexa rank
}

# Get the 32 feature columns (remove F30 duplicate, add Page_Rank)
features_to_keep = list(feature_mapping.values())
# Replace duplicate Country with Page_Rank
features_to_keep = [f if f != "Country" or i < 14 else "Page_Rank" 
                    for i, f in enumerate(features_to_keep)]
features_to_keep = [f for f in features_to_keep if f in df.columns]

# Ensure we have exactly 32 features
if len(features_to_keep) < 32:
    # Add missing features from available columns
    all_cols = [c for c in df.columns if c != "Label" and c not in features_to_keep]
    needed = 32 - len(features_to_keep)
    features_to_keep.extend(all_cols[:needed])
elif len(features_to_keep) > 32:
    # Take first 32
    features_to_keep = features_to_keep[:32]

X = df[features_to_keep].copy()

print(f"\nSelected exactly {len(features_to_keep)} features for training")
print(f"Features: {features_to_keep[:10]}... (showing first 10)")

# Convert ALL features safely to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X.fillna(0, inplace=True)
X = X.to_numpy(dtype=np.float32)

print("\nFinal feature shape:", X.shape)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print("\nClasses:")
for i, c in enumerate(class_names):
    print(f"  {i}: {c}")

# =========================
# SPLIT DATA
# =========================
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp,
    test_size=VAL_RATIO,
    random_state=RANDOM_STATE,
    stratify=y_tmp
)

# reshape labels
y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

train = np.hstack((X_train, y_train))
val   = np.hstack((X_val, y_val))
test  = np.hstack((X_test, y_test))

# =========================
# SAVE OUTPUT
# =========================
np.save("train.npy", train)
np.save("val.npy", val)
np.save("test.npy", test)
np.save("class_names.npy", class_names)

print("\nSaved:")
print(" - train.npy", train.shape)
print(" - val.npy  ", val.shape)
print(" - test.npy ", test.shape)
print(" - class_names.npy")

print(f"\nDone in {(time.time() - start_time):.2f} seconds")
