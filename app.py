import os
import io
import time
import sqlite3
import requests
from datetime import datetime
import random
import copy

import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

try:
    from google.cloud import storage
    from google.oauth2 import service_account
except ModuleNotFoundError:
    storage = None
    service_account = None

# -----------------------------
# Config
# -----------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODE = os.getenv("TROUT_LOCAL_MODE", "1").lower() not in {"0", "false", "no"}
LOCAL_DATA_DIR = os.getenv("TROUT_DATA_DIR", os.path.join(APP_DIR, "local_data"))
LOCAL_MODEL_DIR = os.getenv("TROUT_MODEL_DIR", os.path.join(LOCAL_DATA_DIR, "models"))
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

DB_PATH = os.getenv("TROUT_DB_PATH", os.path.join(LOCAL_DATA_DIR, "feedback.db"))
REVIEW_CSV_PATH = os.getenv(
    "TROUT_REVIEW_CSV_PATH",
    os.getenv("TROUT_CSV_PATH",
    "https://storage.googleapis.com/trout_scale_images/simCLR_endtoend/streamlib.csv"
    )
)
CSV_PATH = REVIEW_CSV_PATH
BASELINE_CKPT_PATH = os.getenv("TROUT_BASELINE_CKPT", os.path.join(LOCAL_MODEL_DIR, "backbone_head.pth"))
METRICS_PATH = os.getenv("TROUT_METRICS_PATH", os.path.join(LOCAL_DATA_DIR, "model_metrics.json"))
EVAL_LOG_PATH = os.getenv("TROUT_EVAL_LOG_PATH", os.path.join(LOCAL_DATA_DIR, "evaluation_log.jsonl"))
VALIDATION_CSV_PATH = os.getenv("TROUT_VALIDATION_CSV_PATH", REVIEW_CSV_PATH)
TEST_CSV_PATH = os.getenv("TROUT_TEST_CSV_PATH", "")

FOLDER_SCAN = None               
NUM_CLASSES = 7
LABEL_NAMES = ["0+", "1+", "2+", "3+", "4+", "5+", "Bad"]
FEEDBACK_TRIGGER = int(os.getenv("TROUT_FEEDBACK_TRIGGER", "20"))
IMPROVEMENT_TOL = float(os.getenv("TROUT_IMPROVEMENT_TOL", "0.0001"))
VALIDATION_BOOTSTRAPS = int(os.getenv("TROUT_VALIDATION_BOOTSTRAPS", "200"))
BACKBONE_UNFREEZE_FEEDBACK_THRESHOLD = int(os.getenv("TROUT_BACKBONE_UNFREEZE_FEEDBACK_THRESHOLD", "300"))
MIN_FEEDBACK_PER_CLASS_FOR_UNFREEZE = int(os.getenv("TROUT_MIN_FEEDBACK_PER_CLASS_FOR_UNFREEZE", "20"))
BACKBONE_LR = float(os.getenv("TROUT_BACKBONE_LR", "1e-5"))

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
CURRENT_MODEL_VERSION = "unloaded"

# -----------------------------
# Local/GCS helpers
# -----------------------------
def get_gcs_client():
    """Return a GCS client only when remote sync is enabled."""
    if LOCAL_MODE:
        return None, None
    if storage is None or service_account is None:
        raise RuntimeError(
            "Google Cloud libraries are not installed. Install google-cloud-storage "
            "or run with TROUT_LOCAL_MODE=1."
        )
    creds_dict = json.loads(st.secrets["gcp"]["credentials"])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(st.secrets["gcp"]["bucket_name"])
    return client, bucket


def latest_local_checkpoint():
    versions = []
    for name in os.listdir(LOCAL_MODEL_DIR):
        if name.startswith("backbone_head_v") and name.endswith(".pth"):
            try:
                num = int(name.replace("backbone_head_v", "").replace(".pth", ""))
                versions.append((num, os.path.join(LOCAL_MODEL_DIR, name)))
            except ValueError:
                pass
    return max(versions, key=lambda x: x[0])[1] if versions else None


def next_local_checkpoint_path():
    latest = latest_local_checkpoint()
    if latest is None:
        next_version = 1
    else:
        latest_name = os.path.basename(latest)
        next_version = int(latest_name.replace("backbone_head_v", "").replace(".pth", "")) + 1
    return os.path.join(LOCAL_MODEL_DIR, f"backbone_head_v{next_version}.pth")

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
    ckpt_path = latest_local_checkpoint()
    if ckpt_path is None and os.path.exists(BASELINE_CKPT_PATH):
        ckpt_path = BASELINE_CKPT_PATH

    if ckpt_path is None and not LOCAL_MODE:
        client, bucket = get_gcs_client()
        baseline_ckpt_key = "simCLR_endtoend/backbone_head.pth"
        if bucket.blob(baseline_ckpt_key).exists(client):
            ckpt_path = BASELINE_CKPT_PATH
            bucket.blob(baseline_ckpt_key).download_to_filename(ckpt_path)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        classifier_head.load_state_dict(ckpt["head_state_dict"])
        CURRENT_MODEL_VERSION = os.path.basename(ckpt_path)
        print(f"🔹 Loaded checkpoint: {CURRENT_MODEL_VERSION}")
    else:
        st.warning(
            "No local checkpoint found. Put backbone_head.pth in local_data/models "
            "or set TROUT_BASELINE_CKPT."
        )
        CURRENT_MODEL_VERSION = "random_untrained"

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
      and 'length' info is merged from REVIEW_CSV_PATH based on filename.
    - If not provided, images are loaded directly from REVIEW_CSV_PATH.
    """
    client, bucket = get_gcs_client()

    # ---------------------------
    # Case 1: Folder-based loading
    # ---------------------------
    if selected_folder:
        try:
            df_ref = pd.read_csv(REVIEW_CSV_PATH, usecols = ["path", "streamlit", "length", "source"])
            df = df_ref[df_ref["path"].astype(str).str.contains(selected_folder, na=False)].copy()
            before = len(df)
            df = df[df["streamlit"] == 1].reset_index(drop = True)
            print(f"Loaded {selected_folder} from CSV: {before} -> {len(df)}")
        except Exception as e:
            if bucket is None:
                st.error(f"Could not load folder from CSV in local mode: {e}")
                return pd.DataFrame(columns=["path", "source", "length"]), []

            prefix = f"troutscales_newimages0825/{selected_folder}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            image_paths = [
                f"https://storage.googleapis.com/{bucket.name}/{b.name}"
                for b in blobs
                if b.name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            df = pd.DataFrame({"path": image_paths})
            df["length"] = None
            df["source"] = "unlabeled"
            df["streamlit"] = 1

        if df.empty:
            st.warning(f"No images found for {selected_folder}")
            return pd.DataFrame(columns=["path", "source", "length"]), []


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
    if REVIEW_CSV_PATH.startswith("http"):
        r = requests.get(REVIEW_CSV_PATH)
        if r.status_code != 200:
            st.error(f"Failed to fetch CSV file: {r.status_code}")
            return pd.DataFrame({"path": []}), []
        df = pd.read_csv(io.StringIO(r.text))
    elif os.path.exists(REVIEW_CSV_PATH):
        df = pd.read_csv(REVIEW_CSV_PATH)
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


def ensure_feedback_schema(con):
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
            model_version TEXT,
            reward REAL,
            used_in_training INTEGER DEFAULT 0,
            trained_model_version TEXT
        )
    """)
    con.commit()

    cur.execute("PRAGMA table_info(feedback);")
    cols = {row[1] for row in cur.fetchall()}
    migrations = {
        "model_version": "ALTER TABLE feedback ADD COLUMN model_version TEXT",
        "reward": "ALTER TABLE feedback ADD COLUMN reward REAL",
        "used_in_training": "ALTER TABLE feedback ADD COLUMN used_in_training INTEGER DEFAULT 0",
        "trained_model_version": "ALTER TABLE feedback ADD COLUMN trained_model_version TEXT",
    }
    for col, ddl in migrations.items():
        if col not in cols:
            cur.execute(ddl)
    con.commit()


def init_db():
    """Initialize local feedback database."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    ensure_feedback_schema(con)
    return con


def upsert_feedback(con, img_path, pred_label, pred_prob, is_correct, correct_label,
                    user="expert", model_version=None):
    """Insert or update expert feedback in the local DB."""
    cur = con.cursor()
    ts = datetime.now().isoformat(timespec="seconds")
    reward = 1.0 if is_correct else -1.0
    cur.execute("""
        INSERT INTO feedback (
            img_path, pred_label, pred_prob, is_correct, correct_label,
            user, ts, model_version, reward, used_in_training
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(img_path) DO UPDATE SET
            pred_label=excluded.pred_label,
            pred_prob=excluded.pred_prob,
            is_correct=excluded.is_correct,
            correct_label=excluded.correct_label,
            user=excluded.user,
            ts=excluded.ts,
            model_version=excluded.model_version,
            reward=excluded.reward,
            used_in_training=0,
            trained_model_version=NULL
    """, (
        img_path, pred_label, pred_prob, is_correct, correct_label,
        user, ts, model_version or CURRENT_MODEL_VERSION, reward
    ))
    con.commit()


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

def load_rgb_image(img_path):
    if str(img_path).startswith("http"):
        r = requests.get(img_path, stream=True, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(img_path).convert("RGB")


def adjust_image_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)


def load_fixed_validation_df():
    """Load a stable labeled validation set used for model selection."""
    if VALIDATION_CSV_PATH.startswith("http"):
        df = pd.read_csv(VALIDATION_CSV_PATH)
    else:
        df = pd.read_csv(VALIDATION_CSV_PATH)

    required = {"path", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Validation CSV must include 'path' and 'label' columns.")

    df = df.dropna(subset=["path", "label"]).copy()
    if "source" in df.columns:
        df = df[df["source"].astype(str).str.lower() == "labeled"].copy()

    # Prefer rows excluded from the expert-review/training stream if the split flag exists.
    if "streamlit" in df.columns and (df["streamlit"] == 0).any():
        df = df[df["streamlit"] == 0].copy()

    return df.reset_index(drop=True)


@torch.no_grad()
def evaluate_on_fixed_validation(model, transform, model_version):
    df_val = load_fixed_validation_df()
    if df_val.empty:
        raise ValueError("No labeled validation rows found.")

    model.eval()
    y_true, y_pred = [], []
    skipped = 0

    for _, row in df_val.iterrows():
        try:
            img = load_rgb_image(row["path"])
            x = transform(img).unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            y_pred.append(pred)
            y_true.append(int(row["label"]))
        except Exception as e:
            skipped += 1
            print(f"Validation skipped {row['path']}: {e}")

    if not y_true:
        raise ValueError("Validation failed: no images could be evaluated.")

    labels = list(range(NUM_CLASSES))
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rng = np.random.default_rng(100)
    boot_scores = []
    if len(y_true_arr) > 1 and VALIDATION_BOOTSTRAPS > 0:
        for _ in range(VALIDATION_BOOTSTRAPS):
            idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
            boot_scores.append(
                f1_score(
                    y_true_arr[idx],
                    y_pred_arr[idx],
                    labels=labels,
                    average="macro",
                    zero_division=0
                )
            )
    if boot_scores:
        ci_low, ci_high = np.quantile(boot_scores, [0.025, 0.975])
    else:
        ci_low = ci_high = macro_f1
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "validation_rows": len(df_val),
        "evaluated_rows": len(y_true),
        "skipped": skipped,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "macro_f1_ci_low": float(ci_low),
        "macro_f1_ci_high": float(ci_high),
        "class_f1": {name: report[name]["f1-score"] for name in LABEL_NAMES},
        "confusion_matrix": cm.tolist(),
    }
    return result


def load_best_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def save_best_metrics(metrics):
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def append_eval_log(metrics, accepted, reason):
    record = dict(metrics)
    record["accepted"] = bool(accepted)
    record["decision_reason"] = reason
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_eval_history():
    if not os.path.exists(EVAL_LOG_PATH):
        return pd.DataFrame()

    records = []
    with open(EVAL_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df_hist = pd.DataFrame(records)
    df_hist["run"] = range(1, len(df_hist) + 1)
    for col in ["macro_f1", "accuracy", "macro_f1_ci_low", "macro_f1_ci_high"]:
        if col in df_hist.columns:
            df_hist[col] = pd.to_numeric(df_hist[col], errors="coerce")
    return df_hist


def candidate_improved(candidate, previous):
    if previous is None:
        return True, "no previous validation metrics"

    cand_macro = float(candidate["macro_f1"])
    prev_macro = float(previous["macro_f1"])
    if cand_macro > prev_macro + IMPROVEMENT_TOL:
        return True, f"macro F1 improved {prev_macro:.4f} -> {cand_macro:.4f}"

    cand_class = candidate.get("class_f1", {})
    prev_class = previous.get("class_f1", {})
    common = [name for name in LABEL_NAMES if name in cand_class and name in prev_class]
    if common:
        no_class_worse = all(
            cand_class[name] >= prev_class[name] - IMPROVEMENT_TOL
            for name in common
        )
        any_class_better = any(
            cand_class[name] > prev_class[name] + IMPROVEMENT_TOL
            for name in common
        )
        if no_class_worse and any_class_better:
            return True, "class F1 improved without lowering any tracked class"

    return False, f"no validation improvement over macro F1 {prev_macro:.4f}"


def feedback_class_counts(con):
    df_counts = pd.read_sql_query(
        """
        SELECT correct_label, COUNT(*) AS n
        FROM feedback
        WHERE is_correct IS NOT NULL
          AND correct_label IS NOT NULL
        GROUP BY correct_label
        """,
        con
    )
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for _, row in df_counts.iterrows():
        label = int(row["correct_label"])
        if label in counts:
            counts[label] = int(row["n"])
    return counts


def feedback_balance_table(con):
    counts = feedback_class_counts(con)
    rows = []
    max_count = max(counts.values()) if counts else 0
    for label_idx, label_name in enumerate(LABEL_NAMES):
        count = counts.get(label_idx, 0)
        rows.append({
            "Label": label_name,
            "Feedback": count,
            "Needed for backbone": max(0, MIN_FEEDBACK_PER_CLASS_FOR_UNFREEZE - count),
            "Underrepresented": count < max_count * 0.5 if max_count > 0 else False,
        })
    return pd.DataFrame(rows)


def validation_distribution_table():
    try:
        df_val = load_fixed_validation_df()
    except Exception:
        return pd.DataFrame()

    rows = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        rows.append({
            "Label": label_name,
            "Validation": int((df_val["label"].astype(int) == label_idx).sum())
        })
    return pd.DataFrame(rows)


def should_unfreeze_backbone(con):
    counts = feedback_class_counts(con)
    total = sum(counts.values())
    min_class_count = min(counts.values()) if counts else 0
    ready = (
        total >= BACKBONE_UNFREEZE_FEEDBACK_THRESHOLD
        and min_class_count >= MIN_FEEDBACK_PER_CLASS_FOR_UNFREEZE
    )
    reason = (
        f"{total}/{BACKBONE_UNFREEZE_FEEDBACK_THRESHOLD} total feedback, "
        f"min class {min_class_count}/{MIN_FEEDBACK_PER_CLASS_FOR_UNFREEZE}"
    )
    return ready, reason


def set_last_resnet_block_trainable(backbone, trainable):
    base = backbone[0] if isinstance(backbone, nn.Sequential) else backbone
    if not hasattr(base, "layer4"):
        return []
    params = list(base.layer4.parameters())
    for p in params:
        p.requires_grad = trainable
    return params


def class_weight_tensor(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = len(labels) / (nonzero.sum() * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def fine_tune_on_feedback(model, transform, con,
                          batch_trigger=FEEDBACK_TRIGGER,
                          batch_size=8, lr=1e-4, epochs=3,
                          replay_ratio=0.5):
    """
    Reinforcement-style online update from expert rewards.

    Correct feedback is treated as reward +1 and uses the predicted label.
    Incorrect feedback is treated as reward -1 and uses the expert label.
    Only the classifier head is updated; the SimCLR backbone remains frozen.
    """
    global CURRENT_MODEL_VERSION
    device = DEVICE

    df_pending = pd.read_sql_query(
        """
        SELECT id, img_path, correct_label, reward
        FROM feedback
        WHERE is_correct IS NOT NULL
          AND correct_label IS NOT NULL
          AND COALESCE(used_in_training, 0) = 0
        ORDER BY id
        """,
        con
    )

    if len(df_pending) < batch_trigger:
        return f"Waiting for more new feedback... ({len(df_pending)}/{batch_trigger})"

    df_batch = df_pending.head(batch_trigger).copy()

    replay_n = int(len(df_batch) * replay_ratio)
    df_replay = pd.read_sql_query(
        """
        SELECT id, img_path, correct_label, reward
        FROM feedback
        WHERE is_correct IS NOT NULL
          AND correct_label IS NOT NULL
          AND COALESCE(used_in_training, 0) = 1
        ORDER BY RANDOM()
        LIMIT ?
        """,
        con,
        params=(replay_n,)
    )

    df_used = pd.concat([df_batch, df_replay], ignore_index=True)
    print(f"Fine-tuning on {len(df_batch)} new feedback + {len(df_replay)} replay samples")

    x_list, y_list = [], []
    for _, row in df_used.iterrows():
        try:
            img = load_rgb_image(row["img_path"])
            x = transform(img)
            x_list.append(x)
            y_list.append(int(row["correct_label"]))
        except Exception as e:
            print(f"⚠️ Skipped {row['img_path']}: {e}")
            continue

    if not x_list:
        return "No valid feedback images found."

    X = torch.stack(x_list).to(device)
    y = torch.tensor(y_list).to(device)
    y_np = np.array(y_list, dtype=int)

    backbone, classifier_head = model[0], model[1]
    previous_version = CURRENT_MODEL_VERSION
    previous_head_state = copy.deepcopy(classifier_head.state_dict())
    previous_backbone_state = copy.deepcopy(backbone.state_dict())

    previous_metrics = load_best_metrics()
    if previous_metrics is None:
        try:
            previous_metrics = evaluate_on_fixed_validation(model, transform, previous_version)
            save_best_metrics(previous_metrics)
            append_eval_log(previous_metrics, accepted=True, reason="initial validation baseline")
        except Exception as e:
            return f"Validation baseline failed; model was not updated. {e}"

    unfreeze_backbone, unfreeze_reason = should_unfreeze_backbone(con)
    backbone_params = set_last_resnet_block_trainable(backbone, unfreeze_backbone)
    if unfreeze_backbone:
        backbone.train()
    else:
        backbone.eval()
    classifier_head.train()
    optimizer_groups = [{"params": classifier_head.parameters(), "lr": lr}]
    if unfreeze_backbone and backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    optimizer = torch.optim.Adam(optimizer_groups)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_np))

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            optimizer.zero_grad()
            if unfreeze_backbone:
                feats = backbone(xb)
            else:
                with torch.no_grad():
                    feats = backbone(xb)
            logits = classifier_head(feats)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {total_loss / len(X):.4f}")

    ckpt_path = next_local_checkpoint_path()
    new_name = os.path.basename(ckpt_path)

    try:
        candidate_metrics = evaluate_on_fixed_validation(model, transform, new_name)
    except Exception as e:
        classifier_head.load_state_dict(previous_head_state)
        backbone.load_state_dict(previous_backbone_state)
        set_last_resnet_block_trainable(backbone, False)
        model.eval()
        return f"Candidate validation failed; restored {previous_version}. {e}"

    accepted, reason = candidate_improved(candidate_metrics, previous_metrics)
    append_eval_log(candidate_metrics, accepted=accepted, reason=reason)

    if not accepted:
        classifier_head.load_state_dict(previous_head_state)
        backbone.load_state_dict(previous_backbone_state)
        set_last_resnet_block_trainable(backbone, False)
        model.eval()
        print(f"Rejected candidate {new_name}: {reason}")
        return (
            f"Candidate rejected: {reason}. "
            f"Best macro F1 remains {previous_metrics['macro_f1']:.4f}."
        )

    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dict": classifier_head.state_dict()
    }, ckpt_path)
    candidate_metrics["checkpoint_path"] = ckpt_path
    candidate_metrics["backbone_unfrozen"] = bool(unfreeze_backbone)
    candidate_metrics["unfreeze_reason"] = unfreeze_reason
    save_best_metrics(candidate_metrics)
    CURRENT_MODEL_VERSION = new_name

    cur = con.cursor()
    pending_ids = df_batch["id"].astype(int).tolist()
    placeholders = ",".join("?" for _ in pending_ids)
    cur.execute(
        f"""
        UPDATE feedback
        SET used_in_training = 1,
            trained_model_version = ?
        WHERE id IN ({placeholders})
        """,
        [new_name] + pending_ids
    )
    cur.execute("""
        UPDATE feedback
        SET pred_label = NULL,
            pred_prob  = NULL,
            model_version = NULL
        WHERE is_correct IS NULL
    """)
    con.commit()
    set_last_resnet_block_trainable(backbone, False)
    model.eval()

    print(f"Accepted candidate {new_name}: {reason}. Saved locally as {ckpt_path}.")
    return (
        f"Accepted {new_name}: {reason}. "
        f"Saved {'head + layer4' if unfreeze_backbone else 'head-only'} update "
        f"({len(df_batch)} new + {len(df_replay)} replay)."
    )

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
st.sidebar.info("Review queue is loaded automatically from review.csv")

if "last_filter" not in st.session_state or st.session_state.last_filter != source_filter:
    st.session_state.idx = 0
    st.session_state.last_filter = source_filter

#show_feedback_table = st.sidebar.checkbox("Show feedback table")

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
df, paths = load_image_list()
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

st.sidebar.subheader("Feedback Balance")
balance_df = feedback_balance_table(con)
st.sidebar.dataframe(balance_df, use_container_width=True, hide_index=True)
ready_unfreeze, unfreeze_status = should_unfreeze_backbone(con)
if ready_unfreeze:
    st.sidebar.success(f"Backbone layer4 updates enabled: {unfreeze_status}")
else:
    st.sidebar.caption(f"Backbone layer4 stays frozen: {unfreeze_status}")

val_dist_df = validation_distribution_table()
if not val_dist_df.empty:
    st.sidebar.subheader("Validation Balance")
    st.sidebar.dataframe(val_dist_df, use_container_width=True, hide_index=True)

# 🔀 Shuffle AFTER filtering, once for the review queue
import random

rand_key = "random_paths_review"
folder_key = "last_review_queue"
review_queue_id = REVIEW_CSV_PATH

if st.session_state.get(folder_key) != review_queue_id:
    paths_all = df["path"].tolist()
    random.shuffle(paths_all)
    st.session_state[rand_key] = paths_all
    st.session_state[folder_key] = review_queue_id
    st.session_state.idx = 0  

if rand_key not in st.session_state:
    paths_all = df["path"].tolist()
    random.shuffle(paths_all)
    st.session_state[rand_key] = paths_all

paths = st.session_state[rand_key]

# ✅ Resume from last feedback in the review queue
idx_key = "idx_review"

fb_df = fetch_all_feedback(con)


if idx_key not in st.session_state:
    st.session_state[idx_key] = 0
    
if "initialized" not in st.session_state or st.session_state.get("last_queue_idx") != review_queue_id:
    st.session_state.initialized = True
    st.session_state["last_queue_idx"] = review_queue_id

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
    filtered_review = len(paths)
    progress_text = f"Review queue | {st.session_state[idx_key] + 1} / {filtered_review} images"

    st.progress(
        (st.session_state[idx_key] + 1) / filtered_review,
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
    contrast_factor = st.slider(
        "Contrast",
        min_value=0.2,
        max_value=3.0,
        value=1.0,
        step=0.05,
        help="Adjusts only the displayed image. Model prediction still uses the original image."
    )
    try:
        display_img = adjust_image_contrast(load_rgb_image(img_path), contrast_factor)
        st.image(display_img)
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
    pred_label, pred_prob, err = predict(model, transform, img_path, con=con)

    # 🔹 If labeled data → show ground truth label
    if source_val == "labeled" and "label" in df.columns:
        label_row = df.loc[df["path"] == img_path, "label"]
        if not label_row.empty:
            true_label = int(label_row.values[0])
            if err:
                st.error(err)
            else:
                st.markdown(
                    f"**Predicted:** `{LABEL_NAMES[pred_label]}` | **Prob:** {pred_prob:.4f} | "
                    f"**Dataset Label:** `{LABEL_NAMES[true_label]}` | **Length:** {length_val} | **Source:** {source_val}"
                )
        else:
            st.warning("Label column missing or empty for this labeled data.")
    else:
        # 🔹 Unlabeled → model prediction
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
            if pred_label is None:
                st.warning("Prediction failed for this image; skip it or fix the image path.")
                st.stop()
            is_correct = 1 if choice == "Correct" else 0
            final_correct = int(pred_label) if is_correct == 1 else int(correct_label)

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
                    user=user_name,
                    model_version=CURRENT_MODEL_VERSION
                )
                st.success("Saved feedback.")

                # 🔹 Check pending expert feedback count
                cur = con.cursor()
                cur.execute("""
                    SELECT COUNT(*)
                    FROM feedback
                    WHERE is_correct IS NOT NULL
                      AND correct_label IS NOT NULL
                      AND COALESCE(used_in_training, 0) = 0
                """)
                feedback_count = cur.fetchone()[0]
                
                # 🔸 Fine-tune after every FEEDBACK_TRIGGER new expert rewards
                if feedback_count >= FEEDBACK_TRIGGER:
                    st.info(f"Fine-tuning triggered automatically (new feedbacks: {feedback_count})")
                    msg = fine_tune_on_feedback(model, transform, con)
                    st.caption(msg)

                    # ✅ Move to next image BEFORE rerun
                    st.session_state[idx_key] = min(len(paths) - 1, st.session_state[idx_key] + 1)
                    st.session_state.idx = st.session_state[idx_key]

                    # ✅ After fine-tuning: reset cache + reload new model
                    st.cache_resource.clear()
                    model, transform, CURRENT_MODEL_VERSION = load_model()
                    st.sidebar.success(f"Reloaded model version: {CURRENT_MODEL_VERSION}")

                    # ✅ Force rerun to refresh sidebar and model state
                    st.rerun()
                else:
                    st.caption(f"Waiting for next fine-tuning... ({feedback_count}/{FEEDBACK_TRIGGER} new feedbacks)")
                
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

st.subheader("Model Performance")
eval_history = load_eval_history()

if eval_history.empty:
    st.caption("No validation history yet. A chart will appear after the first 20-feedback update is evaluated.")
else:
    chart_df = eval_history[["run", "macro_f1", "accuracy"]].dropna(how="all", subset=["macro_f1", "accuracy"])
    st.line_chart(
        chart_df.set_index("run"),
        y=["macro_f1", "accuracy"],
        use_container_width=True
    )

    display_cols = [
        col for col in [
            "run", "model_version", "accepted", "macro_f1", "accuracy",
            "macro_f1_ci_low", "macro_f1_ci_high",
            "evaluated_rows", "skipped", "decision_reason"
        ]
        if col in eval_history.columns
    ]
    st.dataframe(
        eval_history[display_cols].sort_values("run", ascending=False),
        use_container_width=True
    )

    latest_with_cm = eval_history[eval_history["confusion_matrix"].notna()] if "confusion_matrix" in eval_history.columns else pd.DataFrame()
    if not latest_with_cm.empty:
        latest_row = latest_with_cm.iloc[-1]
        cm_df = pd.DataFrame(
            latest_row["confusion_matrix"],
            index=[f"True {name}" for name in LABEL_NAMES],
            columns=[f"Pred {name}" for name in LABEL_NAMES]
        )
        st.caption(f"Latest validation confusion matrix: {latest_row.get('model_version', '')}")
        st.dataframe(cm_df, use_container_width=True)
