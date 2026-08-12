# prepare_dataset_manual.py
import os
import re
import pandas as pd
import requests
import pypdf
import whisper_timestamped as whisper
import torch
import torchaudio
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# -------------------------------------------------------------------
# 0. Read Google Sheet directly as CSV
# -------------------------------------------------------------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AKsEOYGylGEC7fIN24XqUEI1CZXxH22cQcIpbb9eji0/export?format=csv&gid=0"
df = pd.read_csv(SHEET_URL)
print(f"Loaded {len(df)} rows from Google Sheet")
print("Columns found:", df.columns.tolist())

# -------------------------------------------------------------------
# 1. Select columns with priority
# -------------------------------------------------------------------
def find_column(possible_names, df_columns):
    for name in possible_names:
        for col in df_columns:
            if name.lower() in col.lower():
                return col
    return None

mp3_col = find_column(["mp3", "oral", "audio"], df.columns)

# Prefer exact 'Transcript Link', then fallback
if 'Transcript Link' in df.columns:
    transcript_col = 'Transcript Link'
else:
    transcript_col = find_column(["transcript", "pdf"], df.columns)

if 'Case Number' in df.columns:
    case_col = 'Case Number'
else:
    case_col = find_column(["case number", "case no", "case"], df.columns)

if 'Hearing Date' in df.columns:
    date_col = 'Hearing Date'
else:
    date_col = find_column(["hearing date", "date"], df.columns)

if not all([mp3_col, transcript_col, case_col, date_col]):
    raise ValueError("Missing required columns. Check the sheet.")

print(f"Using columns: MP3='{mp3_col}', Transcript='{transcript_col}', Case='{case_col}', Date='{date_col}'")

# -------------------------------------------------------------------
# 2. Download helper with retries and user-agent
# -------------------------------------------------------------------
def download_file(url, dest_path, retries=4, delay=5):
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists – skipping download")
        return True
    if pd.isna(url) or not url:
        print(f"Skipping {dest_path} – no URL")
        return False
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    for attempt in range(retries):
        try:
            print(f"Downloading {url} to {dest_path} (attempt {attempt+1}/{retries})")
            response = requests.get(url, stream=True, timeout=90, headers=headers)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # exponential backoff
            else:
                print(f"Giving up on {url}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
    return False

# -------------------------------------------------------------------
# 3. Parse PDF transcript (unchanged)
# -------------------------------------------------------------------
def parse_pdf_transcript(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception:
        return []
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+$', line):
                continue
            if any(key in line for key in ["CHIEF JUSTICE'S COURT", "SUPREME COURT OF INDIA",
                                           "Document Control", "Transcribed by"]):
                continue
            cleaned.append(line)
        full_text += ' '.join(cleaned)
    
    raw_utterances = []
    current_text = []
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^([A-Z][A-Z\s\.]+):\s*(.*)', line)
        if match:
            if current_text:
                raw_utterances.append(' '.join(current_text))
            current_text = [match.group(2)]
        else:
            if current_text:
                current_text.append(line)
            else:
                raw_utterances.append(line)
    if current_text:
        raw_utterances.append(' '.join(current_text))
    
    final_utterances = []
    for utt in raw_utterances:
        sentences = re.split(r'(?<=[.!?])\s+', utt)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                final_utterances.append(sent)
    return final_utterances

# -------------------------------------------------------------------
# 4. Alignment (unchanged)
# -------------------------------------------------------------------
def align_audio_with_transcript(audio_path, utterances):
    if not utterances or not os.path.exists(audio_path):
        return []
    full_text = ' '.join(utterances)
    model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        result = whisper.align(model, audio_path, full_text, language="en")
    except Exception as e:
        print(f"Alignment error: {e}")
        return []
    
    words = []
    for seg in result.get('segments', []):
        for word in seg.get('words', []):
            words.append({
                'text': word['text'],
                'start': word['start'],
                'end': word['end']
            })
    if not words:
        return []
    
    concat_text = ""
    word_starts = []
    word_ends = []
    for w in words:
        if concat_text:
            concat_text += " "
        start_pos = len(concat_text)
        concat_text += w['text']
        end_pos = len(concat_text)
        word_starts.append(start_pos)
        word_ends.append(end_pos)
    
    segments = []
    def clean(t):
        return re.sub(r'[^a-zA-Z0-9 ]', '', t).lower()
    cleaned_concat = clean(concat_text)
    
    for utt in utterances:
        cleaned_utt = clean(utt)
        start_idx = cleaned_concat.find(cleaned_utt)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(cleaned_utt)
        first_word_idx = None
        last_word_idx = None
        for i, (s, e) in enumerate(zip(word_starts, word_ends)):
            if s >= start_idx and first_word_idx is None:
                first_word_idx = i
            if e <= end_idx:
                last_word_idx = i
        if first_word_idx is not None and last_word_idx is not None:
            start_time = words[first_word_idx]['start']
            end_time = words[last_word_idx]['end']
            if end_time - start_time >= 1.0:
                segments.append((start_time, end_time, utt))
    return segments

# -------------------------------------------------------------------
# 5. Main processing loop
# -------------------------------------------------------------------
os.makedirs("audio", exist_ok=True)
os.makedirs("pdfs", exist_ok=True)

all_segments = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    case_num = row.get(case_col, "")
    if pd.isna(case_num):
        case_num = f"row_{idx}"
    hearing_date = row.get(date_col, "")
    if pd.isna(hearing_date):
        hearing_date = "unknown"
    mp3_url = row.get(mp3_col, "")
    pdf_url = row.get(transcript_col, "")
    
    if pd.isna(mp3_url) or not mp3_url or pd.isna(pdf_url) or not pdf_url:
        print(f"Skipping row {idx}: missing URL")
        continue
    
    safe_case = re.sub(r'[^a-zA-Z0-9]', '_', str(case_num))
    safe_date = re.sub(r'[^a-zA-Z0-9]', '_', str(hearing_date))
    mp3_path = f"audio/{safe_case}_{safe_date}.mp3"
    pdf_path = f"pdfs/{safe_case}_{safe_date}.pdf"
    
    # Download MP3 (usually works)
    mp3_ok = download_file(mp3_url, mp3_path, retries=3)
    # Download PDF – if it fails, we'll still try to process if the file exists (manual placement)
    pdf_ok = download_file(pdf_url, pdf_path, retries=4, delay=5)
    
    # If MP3 missing, skip; if PDF missing, skip (unless you want to test)
    if not os.path.exists(mp3_path) or not os.path.exists(pdf_path):
        print(f"Skipping row {idx}: missing required file(s)")
        continue
    
    utterances = parse_pdf_transcript(pdf_path)
    if not utterances:
        print(f"No utterances for {safe_case}")
        continue
    
    segs = align_audio_with_transcript(mp3_path, utterances)
    for start, end, text in segs:
        all_segments.append({
            "audio_path": mp3_path,
            "start": start,
            "end": end,
            "sentence": text
        })

# -------------------------------------------------------------------
# 6. Create dataset
# -------------------------------------------------------------------
if not all_segments:
    print("No segments created. Check your data and alignment.")
    exit()

dataset = Dataset.from_list(all_segments)

unique_files = dataset.unique("audio_path")
train_files, val_files = train_test_split(unique_files, test_size=0.1, random_state=42)
train_indices = [i for i, row in enumerate(dataset) if row["audio_path"] in train_files]
val_indices = [i for i, row in enumerate(dataset) if row["audio_path"] in val_files]
train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

train_dataset.save_to_disk("supreme_train")
val_dataset.save_to_disk("supreme_eval")

print(f"Train segments: {len(train_dataset)}, Val segments: {len(val_dataset)}")