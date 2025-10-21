import os
import io
import time
import sqlite3
from datetime import datetime

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

DB_PATH = "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/feedback.db"
CSV_PATH = "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/final_results.csv"       
FOLDER_SCAN = None               
NUM_CLASSES = 7
LABEL_NAMES = ["0+", "1+", "2+", "3+", "4+", "5+", "Bad"]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# -----------------------------
# Model Loading (user-provided)
# -----------------------------
@st.cache_resource
def load_model():
    
    set_seed(100)

    backbone = torch.load(
        "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/backbone_resnet18_simclr2.pth",
        map_location=DEVICE, weights_only=False
    )
    backbone = nn.Sequential(backbone, nn.Flatten())

    classifier_head = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES)
    ).to(DEVICE)

    updated_path = "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/classifier_head_updated.pth"
    original_path = "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/classifier_head.pth"
    if os.path.exists(updated_path):
    	print("🔹 Loading fine-tuned classifier head...")
    	state_dict = torch.load(updated_path, map_location=DEVICE)
    else:
    	print("⚪ Loading original classifier head...")
    	state_dict = torch.load(original_path, map_location=DEVICE)
    classifier_head.load_state_dict(state_dict)

    model = nn.Sequential(backbone, classifier_head).to(DEVICE)
    model.eval()

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
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        assert "path" in df.columns, "CSV must have a 'path' column."
        df = df.dropna(subset=["path"]).reset_index(drop=True)
        paths = df["path"].tolist()
    elif FOLDER_SCAN and os.path.isdir(FOLDER_SCAN):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths = []
        for root, _, files in os.walk(FOLDER_SCAN):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        df = pd.DataFrame({"path": paths})
    else:
        df = pd.DataFrame({"path": []})
        paths = []

    return df, paths

# -----------------------------
# SQLite
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
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
    return con

def upsert_feedback(con, img_path, pred_label, pred_prob, is_correct, correct_label, user="expert"):
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

def fetch_all_feedback(con):
    return pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", con)

# -----------------------------
# Inference (with caching)
# -----------------------------
@torch.no_grad()
def predict(model, transform, img_path, con=None):

    # 🔹 1. Check DB cash
    if con is not None:
        cur = con.cursor()
        cur.execute("SELECT pred_label, pred_prob FROM feedback WHERE img_path = ?", (img_path,))
        row = cur.fetchone()
        if row and row[0] is not None:
            # Already predicted result exists → return from cache
            return int(row[0]), float(row[1]), None

    # 🔹 2. conduct new inference 
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, None, f"Image open error: {e}"

    x = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    prob_vals, pred_idx = torch.max(probs, dim=1)

    pred_label = pred_idx.item()
    pred_prob = float(prob_vals.item())

    # 🔹 3. Cache the result in the database (only if no existing label is present)
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

    return pred_label, pred_prob, None
    
    
# -----------------------------
# Online Fine-tuning Function
# -----------------------------
def fine_tune_on_feedback(model, transform, con,
                          batch_size_gpu=8, batch_size_cpu=4,
                          lr_gpu=1e-4, lr_cpu=5e-4,
                          epoch_gpu=3, epoch_cpu=2):
    """
    Fine-tune classifier head when new incorrect feedback accumulates.
    Automatically adjust hyperparameters based on GPU/CPU environment.
    """
    device = DEVICE
    df_fb = pd.read_sql_query(
        "SELECT img_path, correct_label FROM feedback WHERE is_correct=0 AND correct_label IS NOT NULL",
        con
    )
    if len(df_fb) == 0:
        return "🟡 No new incorrect feedback yet."

    # Use only the most recent feedback (limit by batch size).
    if device.type == "cuda":
        batch_size, lr, epochs = batch_size_gpu, lr_gpu, epoch_gpu
    else:
        batch_size, lr, epochs = batch_size_cpu, lr_cpu, epoch_cpu

    df_recent = df_fb.tail(batch_size)

    x_list, y_list = [], []
    for _, row in df_recent.iterrows():
        try:
            img = Image.open(row["img_path"]).convert("RGB")
            x = transform(img)
            x_list.append(x)
            y_list.append(int(row["correct_label"]))
        except Exception:
            continue

    if not x_list:
        return "⚠️ No valid feedback images found."

    X = torch.stack(x_list).to(device)
    y = torch.tensor(y_list).to(device)

    # only classifier head update
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

    # Save the updated weights.
    torch.save(
        classifier_head.state_dict(),
        "/Users/lee/Library/CloudStorage/OneDrive-UniversityofMissouri/fish/simCLR_endtoend/classifier_head_updated.pth"
    )

    return f"✅ Fine-tuned on {len(df_recent)} samples ({epochs} epoch{'s' if epochs>1 else ''}, lr={lr}, device={device.type})"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Expert Labeling App", layout="wide")

st.title("🧩 Expert-in-the-Loop Labeling (SimCLR → Classifier)")
st.caption("Human-in-the-loop QA for model predictions.")

# Sidebar
st.sidebar.header("Settings")
user_name = st.sidebar.text_input("User (optional)", value="expert")

source_filter = st.sidebar.selectbox(
    "Select data source",
    options=["all", "labeled", "unlabeled"],
    index=0,
    format_func=lambda x: x.capitalize()
)

if "last_filter" not in st.session_state or st.session_state.last_filter != source_filter:
    st.session_state.idx = 0
    st.session_state.last_filter = source_filter

show_feedback_table = st.sidebar.checkbox("Show feedback table")

# Load model/data/db
model, transform = load_model()
df, paths = load_image_list()
con = init_db()

# 🔹 Apply filter
if source_filter != "all" and "source" in df.columns:
    df = df[df["source"].str.lower() == source_filter.lower()].reset_index(drop=True)
    paths = df["path"].tolist()

# ✅ Resume from last feedback per source_filter
import os

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
    # 파일명만 표시 (예: cu1_30_06.png)
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

# Optional table
if show_feedback_table:
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

