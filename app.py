import os
import io
import time
import sqlite3
import requests
from datetime import datetime
import random

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
from sklearn.metrics import classification_report, accuracy_score, f1_score

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
CSV_PATH = "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/streamlib.csv"  

FOLDER_SCAN = None               
NUM_CLASSES = 7
LABEL_NAMES = ["0+", "1+", "2+", "3+", "4+", "5+", "Bad"]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# -----------------------------
# Model Version Control
# -----------------------------
def get_next_model_version(bucket):
    """Return the next classifier_head version name based on files in GCS."""
    blobs = list(bucket.list_blobs(prefix="simCLR_endtoend/classifier_head_v"))
    versions = []
    for b in blobs:
        name = os.path.basename(b.name)
        if name.startswith("classifier_head_v") and name.endswith(".pth"):
            try:
                num = int(name.replace("classifier_head_v", "").replace(".pth", ""))
                versions.append(num)
            except ValueError:
                pass
    next_version = max(versions) + 1 if versions else 1
    new_name = f"classifier_head_v{next_version}.pth"
    print(f"🆕 Next model version will be: {new_name}")
    return new_name

# -----------------------------
# Model Loading (user-provided)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load model in the following order:
    1) Latest fine-tuned checkpoint: simCLR_endtoend/backbone_head_v{N}.pth
    2) Baseline checkpoint: simCLR_endtoend/backbone_head.pth
    3) Original classifier head only (first-time fallback)
    """
    global CURRENT_MODEL_VERSION
    set_seed(100)
    client, bucket = get_gcs_client()

    # ---------- Helper: find latest vN checkpoint ----------
    def get_latest_checkpoint_key(bucket):
        blobs = list(bucket.list_blobs(prefix="simCLR_endtoend/backbone_head_v"))
        versions = []
        for b in blobs:
            name = os.path.basename(b.name)
            if name.startswith("backbone_head_v") and name.endswith(".pth"):
                try:
                    num = int(name.replace("backbone_head_v", "").replace(".pth", ""))
                    versions.append((num, b.name))
                except ValueError:
                    pass
        return max(versions, key=lambda x: x[0])[1] if versions else None

    latest_ckpt_key = get_latest_checkpoint_key(bucket)
    baseline_ckpt_key = "simCLR_endtoend/backbone_head.pth"

    # ---------- Build backbone and head (same structure as training) ----------
    from torchvision import models
    print("🧩 Building Sequential-wrapped ResNet18 backbone (matches saved checkpoint).")

    
    base_backbone = models.resnet18()
    base_backbone.fc = nn.Identity()
    backbone = nn.Sequential(base_backbone).to(DEVICE)  # wrap inside Sequential

    classifier_head = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES)
    ).to(DEVICE)

    # ---------- Load checkpoint ----------
    if latest_ckpt_key is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(latest_ckpt_key))
        bucket.blob(latest_ckpt_key).download_to_filename(tmp_path)
        ckpt = torch.load(tmp_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        classifier_head.load_state_dict(ckpt["head_state_dict"])
        CURRENT_MODEL_VERSION = os.path.basename(latest_ckpt_key)
        print(f"🔹 Loaded fine-tuned checkpoint: {CURRENT_MODEL_VERSION}")

    elif bucket.blob(baseline_ckpt_key).exists(client):
        tmp_path = os.path.join(tempfile.gettempdir(), "backbone_head.pth")
        bucket.blob(baseline_ckpt_key).download_to_filename(tmp_path)
        ckpt = torch.load(tmp_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        classifier_head.load_state_dict(ckpt["head_state_dict"])
        CURRENT_MODEL_VERSION = os.path.basename(baseline_ckpt_key)
        print(f"🔹 Loaded baseline checkpoint: {CURRENT_MODEL_VERSION}")

    else:
        # ---------- fallback ----------
        print("⚪ No checkpoints found — using original classifier_head only.")
        head_url = "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/classifier_head.pth"
        head_path = os.path.join(tempfile.gettempdir(), "classifier_head.pth")
        if not os.path.exists(head_path):
            r = requests.get(head_url)
            r.raise_for_status()
            with open(head_path, "wb") as f:
                f.write(r.content)
        state_dict = torch.load(head_path, map_location=DEVICE)
        classifier_head.load_state_dict(state_dict)
        CURRENT_MODEL_VERSION = "classifier_head_original.pth"

    # ---------- Freeze backbone ----------
    for p in backbone.parameters():
        p.requires_grad = False

    # ---------- Combine ----------
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

    # ---------- Sidebar ----------
    print(f"✅ Loaded model version: {CURRENT_MODEL_VERSION}")
    st.sidebar.markdown(
        f"<div style='padding:6px; background-color:#f5f5f5; border-radius:8px;'>"
        f"<b>Current Model Version:</b> <code>{CURRENT_MODEL_VERSION}</code>"
        f"</div>",
        unsafe_allow_html=True
    )

    return model, transform, CURRENT_MODEL_VERSION
    
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
def load_image_list(selected_folder=None):
    """
    Load images either from a CSV file or directly from a GCS folder.

    - If `selected_folder` is provided, images are listed from that folder (unlabeled mode)
      and 'length' info is merged from CSV_PATH based on filename.
    - If not provided, images are loaded directly from CSV_PATH (labeled mode).
    """
    client, bucket = get_gcs_client()

    # ---------------------------
    # Case 1: Folder-based loading (GCS unlabeled)
    # ---------------------------
    if selected_folder:
        prefix = f"troutscales_newimages0825/{selected_folder}/"

        # List all image blobs under selected folder
        blobs = list(bucket.list_blobs(prefix=prefix))
        image_paths = [
            f"https://storage.googleapis.com/{bucket.name}/{b.name}"
            for b in blobs
            if b.name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_paths:
            st.warning(f"No images found in {prefix}")
            return pd.DataFrame(columns=["path", "source", "length"]), []

        # Create base dataframe
        df = pd.DataFrame({
            "path": image_paths
        })
        df["base"] = df["path"].apply(lambda x: os.path.basename(str(x)))

        # ✅ Merge length info from CSV_PATH (by filename)
        try:
            df_ref = pd.read_csv(CSV_PATH, usecols = ["path", "streamlit", "length", "source"])
            df_ref["base"] = df_ref["path"].apply(lambda x: os.path.basename(str(x)))

            df = df.merge(df_ref[["base", "streamlit", "length", "source"]], on = "base", how ="left")

            before = len(df)
            df = df[df["streamlit"] == 1].reset_index(drop = True)
            print(f"Filtered training + unlabeled: {before} -> {len(df)}")

        except Exception as e:
            print(f"Could not merge or filter by streamlit: {e}")
            df["length"] = None
            df["streamlit"] = 1


        # ✅ Randomize order of paths (preserve once per session)
        if "random_paths" not in st.session_state or st.session_state.get("last_folder") != selected_folder:
            paths = df["path"].tolist()
            random.shuffle(paths)
            st.session_state["random_paths"] = paths
            st.session_state["last_folder"] = selected_folder
            print("🔀 Shuffled image order for this folder.")
        else:
            paths = st.session_state["random_paths"]

        return df, paths

    # ---------------------------
    # Case 2: CSV-based loading (labeled)
    # ---------------------------
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

    if "streamlit" in df.columns:
        df = df[df['streamlit'] == 1].reset_index(drop = True)
        print(f"Loaded only streamlit == 1 images: {len(df)} rows")

    else:
        print("No 'streamlit' column found in CSV; loading all images.")

    # Validate structure
    if "path" not in df.columns:
        st.error("CSV must include a 'path' column.")
        return pd.DataFrame({"path": []}), []

    df = df.dropna(subset=["path"]).reset_index(drop=True)

#    if "source" not in df.columns:
#        df["source"] = "labeled"

    # ✅ Also shuffle labeled paths for fairness
    if "random_paths" not in st.session_state or st.session_state.get("last_folder") != "labeled":
        paths = df["path"].tolist()
        random.shuffle(paths)
        st.session_state["random_paths"] = paths
        st.session_state["last_folder"] = "labeled"
        print("🔀 Shuffled labeled image order.")
    else:
        paths = st.session_state["random_paths"]

    return df, paths

# -----------------------------
# Baseline (Original Model Results)
# -----------------------------
baseline = {
    "model_version": "classifier_head_original.pth",
    "accuracy": 0.86,
    "macro_f1": 0.60,
    "per_class": {
        "0": {"f1": 0.88},
        "1": {"f1": 0.86},
        "2": {"f1": 0.56},
        "3": {"f1": 0.40},
        "4": {"f1": 0.20},
        "5": {"f1": 0.33},
        "6": {"f1": 0.96}
    }
}

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
    Also ensures that the table always includes the 'model_version' column.
    """
    client, bucket = get_gcs_client()
    blob = bucket.blob("simCLR_endtoend/feedback.db")
    tmp_path = os.path.join(tempfile.gettempdir(), "feedback.db")

    # ✅ Case 1: If GCS DB exists → download and use it
    if blob.exists(client):
        blob.download_to_filename(tmp_path)
        print("✅ Downloaded feedback.db from GCS.")
    else:
        # ⚪ Case 2: If not found in GCS → reset local cache and create new DB
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
                ts TEXT,
                model_version TEXT
            )
        """)
        con.commit()

        # ✅ Upload new empty DB to GCS immediately
        blob.upload_from_filename(tmp_path)
        print("☁️ Uploaded new feedback.db to GCS.")
        return con

    # ✅ Return SQLite connection to downloaded DB
    con = sqlite3.connect(tmp_path)

    # 🔹 Ensure the schema includes model_version
    cur = con.cursor()
    cur.execute("PRAGMA table_info(feedback);")
    cols = [row[1] for row in cur.fetchall()]
    if "model_version" not in cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN model_version TEXT;")
        con.commit()
        print("✅ Added missing 'model_version' column to feedback table.")
    else:
        print("ℹ️ 'model_version' column already exists.")

    return con


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
# global version tag (update after fine-tune)

@torch.no_grad()
def predict(model, transform, img_path, con=None):
    """
    Predict label and probability for an image.
    Uses database cache only if the stored model_version matches the current one.
    """

    # 🔹 1. Check cache in the database (with version check)
    if con is not None:
        cur = con.cursor()
        cur.execute("""
            SELECT pred_label, pred_prob, model_version 
            FROM feedback 
            WHERE img_path = ?
        """, (img_path,))
        row = cur.fetchone()

        if row and row[0] is not None and row[2] == CURRENT_MODEL_VERSION:
            # ✅ Cache valid (same model version)
            return int(row[0]), float(row[1]), None
        # ⚠️ Cache exists but outdated (different model version)
        elif row and row[2] != CURRENT_MODEL_VERSION:
            print(f"🔄 Recomputing prediction for {os.path.basename(img_path)} (model updated).")

    # 🔹 2. Load image (from URL or local path)
    try:
        if img_path.startswith("http"):
            r = requests.get(img_path, stream=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
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

    # 🔹 4. Save prediction results to DB cache (with current model version)
    if con is not None:
        try:
            cur.execute("""
                INSERT INTO feedback (img_path, pred_label, pred_prob, is_correct, correct_label, user, ts, model_version)
                VALUES (?, ?, ?, NULL, NULL, '', datetime('now'), ?)
                ON CONFLICT(img_path) DO UPDATE SET
                    pred_label=excluded.pred_label,
                    pred_prob=excluded.pred_prob,
                    model_version=excluded.model_version
            """, (img_path, pred_label, pred_prob, CURRENT_MODEL_VERSION))
            con.commit()
        except Exception as e:
            print(f"[Cache insert warning] {e}")

    # 🔹 5. Return prediction result
    return pred_label, pred_prob, None
    

# -----------------------------
# Model Evaluation (with classifier_head_updated)
# -----------------------------
@torch.no_grad()
def evaluate_model(model, transform, df, con=None, upload_to_gcs=True):
    """
    Evaluate current classifier_head_updated.pth model on labeled dataset.
    Uses existing predict() for consistency.
    Saves results to JSON and uploads to GCS.
    """
    # pick only labeled dataset
    df_labeled = df[df["source"] == "labeled"].dropna(subset=["label"])
    if df_labeled.empty:
        return None, "No labeled data found for evaluation."

    y_true, y_pred = [], []
    skipped = 0

    # Prediction
    for _, row in df_labeled.iterrows():
        img_path = row["path"]
        true_label = int(row["label"])
        pred_label, pred_prob, err = predict(model, transform, img_path, con=con)
        if err:
            skipped += 1
            continue
        y_true.append(true_label)
        y_pred.append(pred_label)

    # report table
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": "classifier_head_updated.pth",
        "total_samples": len(df_labeled),
        "skipped": skipped,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report
    }

    # save
    if upload_to_gcs:
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), "evaluation.json")
            with open(tmp_path, "w") as f:
                json.dump(result, f, indent=2)

            client, bucket = get_gcs_client()
            blob = bucket.blob("simCLR_endtoend/eval_logs/evaluation.json")
            blob.upload_from_filename(tmp_path)
            print("☁️ Uploaded evaluation.json to GCS.")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")

    return result, None

# -----------------------------
# Online Fine-tuning Function
# -----------------------------
import tempfile
from google.cloud import storage

def fine_tune_on_feedback(model, transform, con,
                          batch_trigger=20,
                          batch_size=8, lr=1e-4, epochs=3,
                          mix_correct_ratio=0.2):
    """
    Fine-tune only the classifier head every 'batch_trigger' incorrect feedbacks.
    The backbone remains frozen.
    Saves new checkpoints (backbone_head_v{N}.pth) to GCS.
    """
    device = DEVICE

    # 1️⃣ Load feedback
    df_incorrect = pd.read_sql_query(
        "SELECT img_path, correct_label FROM feedback WHERE is_correct=0 AND correct_label IS NOT NULL", con
    )
    df_correct = pd.read_sql_query(
        "SELECT img_path, correct_label FROM feedback WHERE is_correct=1 AND correct_label IS NOT NULL", con
    )

    if len(df_incorrect) == 0:
        return "🟡 No feedback available yet."

    if len(df_incorrect) % batch_trigger != 0:
        return f"⏸️ Waiting for more feedback... ({len(df_incorrect)}/{batch_trigger})"

    n_correct_sample = int(len(df_incorrect) * mix_correct_ratio)
    df_correct_sampled = df_correct.sample(
        n=min(len(df_correct), n_correct_sample), random_state=42
    ) if len(df_correct) > 0 else pd.DataFrame(columns=df_correct.columns)

    df_used = pd.concat([df_incorrect, df_correct_sampled], ignore_index=True)
    print(f"🧩 Fine-tuning on {len(df_used)} samples")

    # 2️⃣ Prepare data
    x_list, y_list = [], []
    for _, row in df_used.iterrows():
        try:
            img = Image.open(requests.get(row["img_path"], stream=True).raw).convert("RGB")
            x = transform(img)
            x_list.append(x)
            y_list.append(int(row["correct_label"]))
        except Exception as e:
            print(f"⚠️ Skipped {row['img_path']}: {e}")
            continue

    if not x_list:
        return "⚠️ No valid feedback images found."

    X = torch.stack(x_list).to(device)
    y = torch.tensor(y_list).to(device)

    # 3️⃣ Fine-tune head only
    backbone, classifier_head = model[0], model[1]
    classifier_head.train()
    optimizer = torch.optim.Adam(classifier_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            optimizer.zero_grad()
            with torch.no_grad():
                feats = backbone(xb)  # no gradient for backbone
            logits = classifier_head(feats)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {total_loss / len(X):.4f}")

    # 4️⃣ Save updated head + frozen backbone to GCS
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save({
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": classifier_head.state_dict()
        }, tmp.name)
        tmp_path = tmp.name

    # Upload to GCS
    storage_client = storage.Client.from_service_account_info(creds_dict)
    bucket = storage_client.bucket(st.secrets["gcp"]["bucket_name"])
    blobs = list(bucket.list_blobs(prefix="simCLR_endtoend/backbone_head_v"))

    versions = []
    for b in blobs:
        name = os.path.basename(b.name)
        if name.startswith("backbone_head_v") and name.endswith(".pth"):
            try:
                versions.append(int(name.replace("backbone_head_v", "").replace(".pth", "")))
            except ValueError:
                pass

    next_version = max(versions) + 1 if versions else 1
    new_name = f"backbone_head_v{next_version}.pth"
    blob = bucket.blob(f"simCLR_endtoend/{new_name}")
    blob.upload_from_filename(tmp_path)

    # Update DB and current version
 
    global CURRENT_MODEL_VERSION
    CURRENT_MODEL_VERSION = new_name

    # keep feedback but renew cache
    cur = con.cursor()
    cur.execute("""
        UPDATE feedback
        SET pred_label = NULL,
            pred_prob  = NULL,
            model_version = NULL
        WHERE is_correct IS NULL
    """)
    con.commit()

    print(f"☁️ Uploaded fine-tuned head as {new_name}.")
    return f"✅ Fine-tuned (head only) and saved as {new_name} ({len(df_used)} samples)."

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
source_filter = 1
st.sidebar.info("Evaluate unlabeled data only")

if "last_filter" not in st.session_state or st.session_state.last_filter != source_filter:
    st.session_state.idx = 0
    st.session_state.last_filter = source_filter

#show_feedback_table = st.sidebar.checkbox("Show feedback table")

# -----------------------------
# Image Folder Selection
# -----------------------------
st.sidebar.subheader("Select Image Folder")

# Available subfolders in GCS
available_folders = [
    "cu1images", "cu2images", "cu3images", "cu4images",
    "du1images", "du2images", "du3images", "du4images",
    "po1images", "po2images", "po3images", "po4images",
    "to1images", "to2images", "to3images", "to4images",
    "tu1images", "tu2images", "tu3images", "tu4images"
]

# Dropdown to choose the folder
selected_folder = st.sidebar.selectbox("Choose a subfolder", available_folders)

# Set FOLDER_SCAN dynamically
FOLDER_SCAN = f"trout_scale_images/troutscales_newimages0825/{selected_folder}/"


st.sidebar.header("Dataset Details")
#    **Labeled Dataset:** 1,393  
#    **Unlabeled Dataset:** 14,828  
st.sidebar.markdown(
    """
    **Total Dataset:** 16,221  
    **Test Dataset:** 2,118  
    **Train Dataset:** 14,103  
    """
)

# Evaluation Example
selected_eval = {
    "version": "Eval_V1",
    "accuracy": 0.88,
    "macro_f1": 0.63,
    "per_class": {
        "0": {"f1": 0.91},
        "1": {"f1": 0.87},
        "2": {"f1": 0.59},
        "3": {"f1": 0.43},
        "4": {"f1": 0.27},
        "5": {"f1": 0.34},
        "6": {"f1": 0.97}
    }
}

# create comparison table
metrics = ["Accuracy", "Macro F1"] + [f"{i}+" for i in range(7)]
baseline_vals = [baseline["accuracy"], baseline["macro_f1"]] + [baseline["per_class"][str(i)]["f1"] for i in range(7)]
eval_vals = [selected_eval["accuracy"], selected_eval["macro_f1"]] + [selected_eval["per_class"][str(i)]["f1"] for i in range(7)]

df_compare = pd.DataFrame({
    "Metric": metrics,
    "Original": baseline_vals,
    selected_eval["version"]: eval_vals
})

# Load model/data/db
model, transform, CURRENT_MODEL_VERSION = load_model()
df, paths = load_image_list(selected_folder=selected_folder)
con = init_db()

# Ensure model_version column exists (for version-aware cache)
def ensure_model_version_column(con):
    """Ensure model_version column exists in feedback table."""
    cur = con.cursor()
    cur.execute("PRAGMA table_info(feedback);")
    cols = [row[1] for row in cur.fetchall()]
    if "model_version" not in cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN model_version TEXT;")
        con.commit()
        print("✅ Added 'model_version' column to feedback table.")
    else:
        print("ℹ️ 'model_version' column already exists.")

# ✅ Add this line
ensure_model_version_column(con)

# 🔀 Shuffle AFTER filtering, once per folder
import random

rand_key = f"random_paths_{selected_folder}"
folder_key = "last_folder"

if st.session_state.get(folder_key) != selected_folder:
    paths_all = df["path"].tolist()
    random.shuffle(paths_all)
    st.session_state[rand_key] = paths_all
    st.session_state[folder_key] = selected_folder
    st.session_state.idx = 0  

if rand_key not in st.session_state:
    paths_all = df["path"].tolist()
    random.shuffle(paths_all)
    st.session_state[rand_key] = paths_all

paths = st.session_state[rand_key]

# ✅ Resume from last feedback per source_filter
# ✅ Resume from last feedback per folder (instead of global)
# 폴더별 index 키를 별도로 유지
idx_key = f"idx_{selected_folder}"

# 데이터베이스에서 이전 피드백 불러오기
fb_df = fetch_all_feedback(con)

# 폴더별 인덱스가 없으면 0으로 초기화
if idx_key not in st.session_state:
    st.session_state[idx_key] = 0
    
if "initialized" not in st.session_state or st.session_state.get("last_folder_idx") != selected_folder:
    st.session_state.initialized = True
    st.session_state["last_folder_idx"] = selected_folder

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
                st.session_state[idx_key] = min(last_idx + 1, len(paths) - 1)
            except StopIteration:
                st.session_state[idx_key] = 0

st.session_state.idx = st.session_state[idx_key]

# Read the last feedback from the database (shared/common)
fb_df = fetch_all_feedback(con)

if idx_key not in st.session_state:
    st.session_state[idx_key] = 0  

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
                st.session_state[idx_key] = min(last_idx + 1, len(paths) - 1)
            except StopIteration:
                st.session_state[idx_key] = 0

# 🔹 Load the index corresponding to the current filter.
st.session_state.idx = st.session_state.get(idx_key, 0)

# Handle empty set
if len(paths) == 0:
    st.warning("No images found. Provide 'to_review.csv' with a 'path' column or set FOLDER_SCAN.")
    st.stop()

# Clamp index
st.session_state.idx = max(0, min(st.session_state.idx, len(paths)-1))

# -----------------------------
# 📊 Progress Bar 
# -----------------------------
try:
    # ✅ streamlit == 1 
    filtered_in_folder = len(paths)
    
    df_ref = pd.read_csv(CSV_PATH, usecols=["path", "streamlit"])
    df_ref = df_ref[df_ref["path"].str.contains(selected_folder, na=False)]
    total_in_folder = len(df_ref[df_ref["streamlit"] == 1])

    progress_text = (
        f"📁 {selected_folder} | "
        f"{st.session_state[idx_key] + 1} / {total_in_folder} images"
        f"(Including testset: {filtered_in_folder})"
    )

    st.progress(
        (st.session_state[idx_key] + 1) / filtered_in_folder,
        text=progress_text
    )
except Exception as e:
    st.progress((st.session_state.idx + 1) / len(paths),
                text=f"{st.session_state.idx+1} / {len(paths)}")
    print(f"⚠️ Progress bar info failed: {e}")

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

    # 🔹 If labeled data → show ground truth label
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
        # 🔹 Unlabeled → model prediction
        pred_label, pred_prob, err = predict(model, transform, img_path, con=con)
        if err:
            st.error(err)
        else:
            st.markdown(
                f"**Predicted:** `{LABEL_NAMES[pred_label]}` | **Prob:** {pred_prob:.4f} | **Length:** {length_val} | **Source:** {source_val}"
            )

    # 🔸 Expert Feedback UI
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

    # 🔹 Navigation buttons
    cols = st.columns(3)

    # ⬅️ Previous
    with cols[0]:
        if st.button("⬅️ Previous"):
            st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
            st.session_state.idx = st.session_state[idx_key]
            st.rerun()

    # ✅ Save & Next
    with cols[1]:
        if st.button("✅ Save & Next"):
            is_correct = 1 if choice == "Correct" else 0
            final_correct = None if is_correct == 1 else int(correct_label)

            if (is_correct == 0) and (final_correct is None):
                st.warning("Please choose a correct label when marking as Incorrect.")
            else:
                # 🔹 Save feedback to DB
                upsert_feedback(
                    con,
                    img_path=img_path,
                    pred_label=int(pred_label),
                    pred_prob=float(pred_prob) if pred_prob is not None else 1.0,
                    is_correct=is_correct,
                    correct_label=final_correct,
                    user=user_name
                )
                st.success("Saved feedback.")

                # 🔹 Check incorrect feedback count
                cur = con.cursor()
                cur.execute("SELECT COUNT(*) FROM feedback WHERE is_correct=0 AND correct_label IS NOT NULL")
                feedback_count = cur.fetchone()[0]

                # 🔸 Fine-tune after every 20 incorrect feedbacks
                if feedback_count % 20 == 0 and feedback_count > 0:
                    st.info(f"🧠 Fine-tuning triggered automatically (feedbacks: {feedback_count})")
                    msg = fine_tune_on_feedback(model, transform, con)
                    st.caption(msg)

                    # ✅ After fine-tuning: reset cache + reload new model
                    st.cache_resource.clear()
                    model, transform, CURRENT_MODEL_VERSION = load_model()
                    st.sidebar.success(f"🔄 Reloaded model version: {CURRENT_MODEL_VERSION}")

                    # ✅ Force rerun to refresh sidebar and model state
                    st.rerun()

                else:
                    st.caption(f"Waiting for next fine-tuning... ({feedback_count % 20}/20 accumulated)")

                # 🔹 Move to next image
                time.sleep(0.1)
                st.session_state[idx_key] = min(len(paths) - 1, st.session_state[idx_key] + 1)
                st.session_state.idx = st.session_state[idx_key]
                st.rerun()

    # ➡️ Skip
    with cols[2]:
        if st.button("Skip ➡️"):
            st.session_state[idx_key] = min(len(paths) - 1, st.session_state[idx_key] + 1)
            st.session_state.idx = st.session_state[idx_key]
            st.rerun()
            
# ----------------------------------------------------
# 📊 Evaluate Fine-Tuned Model (labeled dataset only)
# ----------------------------------------------------
st.divider()
#st.subheader("🔍 Evaluate Fine-Tuned Model (Labeled Dataset)")

#if st.button("Run Evaluation"):
#    with st.spinner("Evaluating fine-tuned model on labeled data..."):
#        df_full, _ = load_image_list()
#        eval_result, err = evaluate_model(model, transform, df_full, con=con)
#        
#        if err:
#            st.error(err)
#        else:
#            st.success(f"✅ Evaluation complete — Accuracy: {eval_result['accuracy']:.3f}, Macro F1: {eval_result['macro_f1']:.3f}")
#            st.dataframe(
#                pd.DataFrame(eval_result["report"]).transpose().round(3),
#                use_container_width=True
#            )
#            st.caption("☁️ Results uploaded to: `simCLR_endtoend/eval_logs/evaluation.json`")

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

