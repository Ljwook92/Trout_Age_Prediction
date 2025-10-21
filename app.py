import os
import io
import time
import sqlite3
import requests
from datetime import datetime

from google.cloud import storage
from google.oauth2 import service_account
import json
import tempfile

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# -----------------------------
# Config
# -----------------------------

# Load from Streamlit secrets
creds_dict = json.loads(st.secrets["gcp"]["credentials"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

client = storage.Client(credentials=credentials)
bucket_name = st.secrets["gcp"]["bucket_name"]
bucket = client.bucket(bucket_name)

DB_PATH = os.path.join(tempfile.gettempdir(), "feedback.db")   
CSV_PATH = "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/final_results.csv"  

FOLDER_SCAN = None               
NUM_CLASSES = 7
LABEL_NAMES = ["0+", "1+", "2+", "3+", "4+", "5+", "Bad"]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# -----------------------------
# Model Loading (user-provided)
# -----------------------------
@st.cache_resource

@st.cache_resource
def load_model():
    """Load SimCLR backbone and classifier head, preferring updated version from GCS"""
    set_seed(100)
    client, bucket = get_gcs_client()

    # ---------- Utility to download file ----------
    def download_if_needed(url, local_path):
        if not os.path.exists(local_path):
            print(f"📥 Downloading {os.path.basename(local_path)} from {url} ...")
            r = requests.get(url)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Saved to {local_path}")
            else:
                raise RuntimeError(f"❌ Failed to download {url} (status {r.status_code})")
        return local_path

    # ---------- Backbone ----------
    backbone_path = download_if_needed(
        "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/backbone_resnet18_simclr2.pth",
        os.path.join(tempfile.gettempdir(), "backbone_resnet18_simclr2.pth")
    )

    # ---------- Classifier Head (check GCS first) ----------
    blob = bucket.blob("simCLR_endtoend/classifier_head_updated.pth")
    if blob.exists(client):
        print("🔹 Found updated classifier on GCS — downloading...")
        tmp_path = os.path.join(tempfile.gettempdir(), "classifier_head_updated.pth")
        blob.download_to_filename(tmp_path)
        head_path = tmp_path
    else:
        print("⚪ No updated classifier found. Using original head.")
        head_path = download_if_needed(
            "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/classifier_head.pth",
            os.path.join(tempfile.gettempdir(), "classifier_head.pth")
        )

    # ---------- Load model ----------
    backbone = torch.load(backbone_path, map_location=DEVICE)
    backbone = nn.Sequential(backbone, nn.Flatten())

    classifier_head = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES)
    ).to(DEVICE)

    state_dict = torch.load(head_path, map_location=DEVICE)
    classifier_head.load_state_dict(state_dict)

    model = nn.Sequential(backbone, classifier_head).to(DEVICE)
    model.eval()

    # ---------- Transform ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, transform

def set_seed(seed=100):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Data Source
# -----------------------------
def load_image_list():
    if CSV_PATH.startswith("http"):  
        r = requests.get(CSV_PATH)
        if r.status_code != 200:
            st.error(f"Failed to fetch CSV file: {r.status_code}")
            return pd.DataFrame({"path": []}), []
        df = pd.read_csv(io.StringIO(r.text))
    elif os.path.exists(CSV_PATH):  
        df = pd.read_csv(CSV_PATH)
    else:
        st.error("CSV path does not exist.")
        return pd.DataFrame({"path": []}), []

    # 기본 구조 확인
    if "path" not in df.columns:
        st.error("CSV must include a 'path' column.")
        return pd.DataFrame({"path": []}), []

    df = df.dropna(subset=["path"]).reset_index(drop=True)
    paths = df["path"].tolist()
    return df, paths

# -----------------------------
# SQLite
# -----------------------------

def get_gcs_client():
    """Load credentials from Streamlit secrets and return GCS client + bucket"""
    creds_dict = json.loads(st.secrets["gcp"]["credentials"])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(st.secrets["gcp"]["bucket_name"])
    return client, bucket


def init_db():
    """Initialize feedback database.
    - If feedback.db exists in GCS → download it.
    - If not, remove any cached local DB, create a new one, and upload it to GCS.
    """
    client, bucket = get_gcs_client()
    blob = bucket.blob("simCLR_endtoend/feedback.db")
    tmp_path = os.path.join(tempfile.gettempdir(), "feedback.db")

    # ✅ Case 1: If GCS DB exists → download and use it
    if blob.exists(client):
        blob.download_to_filename(tmp_path)
        print("✅ Downloaded feedback.db from GCS.")

    # ⚪ Case 2: If not found in GCS → reset local cache and create new DB
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("🗑️ Deleted old local feedback.db cache.")

        print("⚪ No feedback.db found in GCS. Creating new one...")
        con = sqlite3.connect(tmp_path)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                img_path TEXT UNIQUE,
                pred_label INTEGER,
                pred_prob REAL,
                is_correct INTEGER,
                correct_label INTEGER,
                user TEXT,
                ts TEXT
            )
        """)
        con.commit()

        # ✅ Upload new empty DB to GCS immediately
        blob.upload_from_filename(tmp_path)
        print("☁️ Uploaded new feedback.db to GCS.")
        return con

    # ✅ Return SQLite connection to downloaded DB
    return sqlite3.connect(tmp_path)


def upsert_feedback(con, img_path, pred_label, pred_prob, is_correct, correct_label, user="expert"):
    """Insert or update a record, then sync the DB back to GCS"""
    cur = con.cursor()
    ts = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO feedback (img_path, pred_label, pred_prob, is_correct, correct_label, user, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(img_path) DO UPDATE SET
            pred_label=excluded.pred_label,
            pred_prob=excluded.pred_prob,
            is_correct=excluded.is_correct,
            correct_label=excluded.correct_label,
            user=excluded.user,
            ts=excluded.ts
    """, (img_path, pred_label, pred_prob, is_correct, correct_label, user, ts))
    con.commit()

    # Upload updated DB to GCS
    client, bucket = get_gcs_client()
    blob = bucket.blob("simCLR_endtoend/feedback.db")
    tmp_path = con.execute("PRAGMA database_list").fetchone()[2]
    blob.upload_from_filename(tmp_path)
    print("☁️ Uploaded updated feedback.db to GCS.")


def fetch_all_feedback(con):
    """Read all feedback records"""
    return pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", con)

# -----------------------------
# Inference (with caching)
# -----------------------------
@torch.no_grad()
def predict(model, transform, img_path, con=None):
    # 🔹 1. Check cache in the database
    if con is not None:
        cur = con.cursor()
        cur.execute("SELECT pred_label, pred_prob FROM feedback WHERE img_path = ?", (img_path,))
        row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0]), float(row[1]), None

    # 🔹 2. Load image (from URL or local path)
    try:
        if img_path.startswith("http"):
            # If the path is a URL → fetch the image using requests
            r = requests.get(img_path, stream=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            # If the path is local → open directly
            img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, None, f"Image open error: {e}"

    # 🔹 3. Perform inference
    x = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    prob_vals, pred_idx = torch.max(probs, dim=1)

    pred_label = pred_idx.item()
    pred_prob = float(prob_vals.item())

    # 🔹 4. Save prediction results to DB cache (if not already stored)
    if con is not None:
        try:
            cur.execute("""
                INSERT INTO feedback (img_path, pred_label, pred_prob, is_correct, correct_label, user, ts)
                VALUES (?, ?, ?, NULL, NULL, '', datetime('now'))
                ON CONFLICT(img_path) DO UPDATE SET
                    pred_label=excluded.pred_label,
                    pred_prob=excluded.pred_prob
            """, (img_path, pred_label, pred_prob))
            con.commit()
        except Exception as e:
            print(f"[Cache insert warning] {e}")

    # 🔹 5. Return predicted label, probability, and error message (if any)
    return pred_label, pred_prob, None
    
    
# -----------------------------
# Online Fine-tuning Function
# -----------------------------
import tempfile
from google.cloud import storage

def fine_tune_on_feedback(model, transform, con,
                          batch_size_gpu=8, batch_size_cpu=4,
                          lr_gpu=1e-4, lr_cpu=5e-4,
                          epoch_gpu=3, epoch_cpu=1):
    """
    Fine-tune the classifier head using newly corrected feedback samples.
    Automatically adjusts hyperparameters depending on GPU/CPU environment.
    After fine-tuning, saves updated model weights and uploads them to GCS.
    """
    device = DEVICE
    df_fb = pd.read_sql_query(
        "SELECT img_path, correct_label FROM feedback WHERE is_correct=0 AND correct_label IS NOT NULL",
        con
    )
    if len(df_fb) == 0:
        return "🟡 No new incorrect feedback yet."

    # Select hyperparameters based on environment
    if device.type == "cuda":
        batch_size, lr, epochs = batch_size_gpu, lr_gpu, epoch_gpu
    else:
        batch_size, lr, epochs = batch_size_cpu, lr_cpu, epoch_cpu

    # Use only the most recent batch of feedback samples
    df_recent = df_fb.tail(batch_size)

    x_list, y_list = [], []
    for _, row in df_recent.iterrows():
        try:
            # Load image (from URL or local)
            if row["img_path"].startswith("http"):
                r = requests.get(row["img_path"], stream=True)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                img = Image.open(row["img_path"]).convert("RGB")

            x = transform(img)
            x_list.append(x)
            y_list.append(int(row["correct_label"]))
        except Exception as e:
            print(f"⚠️ Skipped: {row['img_path']} ({e})")
            continue

    if not x_list:
        return "⚠️ No valid feedback images found."

    X = torch.stack(x_list).to(device)
    y = torch.tensor(y_list).to(device)

    # Only update the classifier head
    classifier_head = model[-1]
    classifier_head.train()

    optimizer = torch.optim.Adam(classifier_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier_head(model[0](X))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # ✅ Save updated classifier weights locally and upload to GCS
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name
            torch.save(classifier_head.state_dict(), tmp_path)
            print(f"✅ Saved fine-tuned weights locally: {tmp_path}")

        # Initialize GCS client and upload
        storage_client = storage.Client.from_service_account_info(creds_dict)
        bucket = storage_client.bucket(st.secrets["gcp"]["bucket_name"])
        blob = bucket.blob("simCLR_endtoend/classifier_head_updated.pth")
        blob.upload_from_filename(tmp_path)
        print("☁️ Uploaded updated classifier to GCS (overwrite).")
    except Exception as e:
        print(f"⚠️ Upload to GCS failed: {e}")
        return f"⚠️ Fine-tuning done locally but upload failed ({e})"

    return f"✅ Fine-tuned on {len(df_recent)} samples ({epochs} epoch{'s' if epochs>1 else ''}, lr={lr}, device={device.type}) and uploaded to GCS."

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Expert Labeling App", layout="wide")

st.title("🧩 Expert-in-the-Loop Labeling (SimCLR → Classifier)")
st.caption("Human-in-the-loop QA for model predictions.")

# Sidebar
st.sidebar.header("Settings")
user_name = st.sidebar.text_input("User (optional)", value="expert")

# Force 'unlabeled' mode (no sidebar select)
source_filter = "unlabeled"
st.sidebar.info("Data source fixed to 'unlabeled'")

if "last_filter" not in st.session_state or st.session_state.last_filter != source_filter:
    st.session_state.idx = 0
    st.session_state.last_filter = source_filter

#show_feedback_table = st.sidebar.checkbox("Show feedback table")

st.sidebar.header("Dataset Details")

# Load model/data/db
model, transform = load_model()
df, paths = load_image_list()
con = init_db()

# 🔹 Apply filter
if source_filter != "all" and "source" in df.columns:
    df = df[df["source"].str.lower() == source_filter.lower()].reset_index(drop=True)
    paths = df["path"].tolist()

# ✅ Resume from last feedback per source_filter
# Set state key by source type.
filter_key = f"idx_{source_filter}"

# Read the last feedback from the database (shared/common)
fb_df = fetch_all_feedback(con)

if filter_key not in st.session_state:
    st.session_state[filter_key] = 0  # 기본값

# 🔹 Automatic restoration based on the feedback database.
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    if not fb_df.empty and len(paths) > 0:
        current_basenames = {os.path.basename(str(p)) for p in paths}

        fb_df = fb_df.copy()
        fb_df["base"] = fb_df["img_path"].apply(lambda x: os.path.basename(str(x)))
        fb_match = fb_df[fb_df["base"].isin(current_basenames)]

        if not fb_match.empty:
            last_base = fb_match.iloc[0]["base"]
            try:
                last_idx = next(
                    i for i, p in enumerate(paths)
                    if os.path.basename(str(p)) == last_base
                )
                st.session_state[filter_key] = min(last_idx + 1, len(paths) - 1)
            except StopIteration:
                st.session_state[filter_key] = 0

# 🔹 Load the index corresponding to the current filter.
st.session_state.idx = st.session_state.get(filter_key, 0)

# Handle empty set
if len(paths) == 0:
    st.warning("No images found. Provide 'to_review.csv' with a 'path' column or set FOLDER_SCAN.")
    st.stop()

# Clamp index
st.session_state.idx = max(0, min(st.session_state.idx, len(paths)-1))

# Progress
st.progress((st.session_state.idx + 1) / len(paths), text=f"{st.session_state.idx+1} / {len(paths)}")

# Current image
img_path = paths[st.session_state.idx]
left, right = st.columns([1, 1])

with left:
    st.subheader("Image")
    st.text(os.path.basename(img_path))
    try:
        st.image(img_path)
    except Exception as e:
        st.error(f"Cannot display image: {e}")
        
img_path = paths[st.session_state.idx]

length_val = None
if "length" in df.columns:
    length_row = df.loc[df["path"] == img_path, "length"]
    if not length_row.empty:
        length_val = length_row.values[0]
        
source_val = None
if "source" in df.columns:
    source_row = df.loc[df["path"] == img_path, "source"]
    if not source_row.empty:
        source_val = str(source_row.values[0])
                
with right:
    st.subheader("Model Prediction")
    # labeled 
    if source_val == "labeled" and "label" in df.columns:
        label_row = df.loc[df["path"] == img_path, "label"]
        if not label_row.empty:
            true_label = int(label_row.values[0])
            st.markdown(
                f"**Label (from dataset):** `{LABEL_NAMES[true_label]}` | **Length:** {length_val} | **Source:** {source_val}"
            )
        else:
            st.warning("Label column missing or empty for this labeled data.")
    else:
        # Unlabeled
        pred_label, pred_prob, err = predict(model, transform, img_path, con=con)
        if err:
            st.error(err)
        else:
            st.markdown(
                f"**Predicted:** `{LABEL_NAMES[pred_label]}` | **Prob:** {pred_prob:.4f} | **Length:** {length_val} | **Source:** {source_val}"
            )

    st.divider()
    st.subheader("Expert Feedback")
    choice = st.radio(
        "Is the prediction correct?",
        options=["Correct", "Incorrect"],
        index=0,
        horizontal=True
    )

    correct_label = None
    if choice == "Incorrect":
        correct_label = st.selectbox(
            "Select the correct label",
            options=list(range(NUM_CLASSES)),
            format_func=lambda i: f"{i} ({LABEL_NAMES[i]})"
        )

    cols = st.columns(3)
    with cols[0]:
         if st.button("⬅️ Previous"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
            st.session_state[f"idx_{source_filter}"] = st.session_state.idx  
            st.rerun()

    with cols[1]:
        if st.button("✅ Save & Next"):
            is_correct = 1 if choice == "Correct" else 0
            final_correct = None if is_correct == 1 else int(correct_label)

            if (is_correct == 0) and (final_correct is None):
                st.warning("Please choose a correct label when marking as Incorrect.")
            else:
                upsert_feedback(
                    con,
                    img_path=img_path,
                    pred_label=int(pred_label) if source_val != "labeled" else int(true_label),
                    pred_prob=float(pred_prob) if source_val != "labeled" else 1.0,
                    is_correct=is_correct,
                    correct_label=final_correct,
                    user=user_name
                )
                st.success("Saved.")
                msg = fine_tune_on_feedback(model, transform, con)
                st.caption(msg)
                
                time.sleep(0.1)
                st.session_state.idx = min(len(paths)-1, st.session_state.idx + 1)
                st.session_state[f"idx_{source_filter}"] = st.session_state.idx
                st.rerun()

    with cols[2]:
         if st.button("Skip ➡️"):
            st.session_state.idx = min(len(paths)-1, st.session_state.idx + 1)
            st.session_state[f"idx_{source_filter}"] = st.session_state.idx  
            st.rerun()
            
st.divider()

# Always show feedback table (no checkbox)
st.subheader("Saved Feedback")
fb = fetch_all_feedback(con)

fb["img_path"] = fb["img_path"].apply(
    lambda x: f"{os.path.basename(os.path.dirname(str(x)))}/{os.path.basename(str(x))}"
)

st.dataframe(fb, use_container_width=True)

# Export
st.download_button(
    label="📥 Export feedback as CSV",
    data=fetch_all_feedback(con).to_csv(index=False),
    file_name="feedback_export.csv",
    mime="text/csv"
)

