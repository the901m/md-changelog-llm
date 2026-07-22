#!/usr/bin/env python3
import os
import sys
import re
import sqlite3
import hashlib
import time
import signal
import subprocess
import difflib
import json
import urllib.request
from datetime import datetime

# --- Configuration & Paths ---
TARGET_DIR = ""
CHANGELOG_FILE = ""
DB_FILE = ""
LLAMA_SERVER_DIR = ""
LLAMA_SERVER_CMD = ["llama-server", "-m", "/home/user/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-gguf/blobs/3646b4ccd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd", "--mmproj", "/home/user/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-gguf/blobs/58c187648007ca92bd5678b87e862c38794017deb945feea2cf256195e96a"]
LLAMA_URL = "http://localhost:8080/v1/chat/completions"

DRY_RUN = "--dry-run" in sys.argv or "-d" in sys.argv

os.makedirs(os.path.dirname(CHANGELOG_FILE), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_snapshots (
            path TEXT PRIMARY KEY,
            content TEXT,
            hash TEXT
        )
    ''')
    conn.commit()
    return conn

def calculate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_current_md_files(root_dir):
    md_files = {}
    for root, _, files in os.walk(root_dir):
        if ".trash" in root or "Logs" in root or "workspace.json" in root:
            continue
        for file in files:
            if file.endswith(".md"):
                abs_path = os.path.abspath(os.path.join(root, file))
                rel_path = os.path.relpath(abs_path, root_dir)
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    md_files[rel_path] = {
                        "content": content,
                        "hash": calculate_hash(content)
                    }
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")
    return md_files

# --- Clean Diff Extractor ---
def extract_clean_diff(old_content, new_content):
    """Produces unambiguous diff lines, pairing consecutive removals and additions as modifications."""
    diff = difflib.unified_diff(
        old_content.splitlines(), new_content.splitlines(), lineterm="", n=1
    )
    
    diff_lines = []
    pending_removals = []
    
    for line in diff:
        if line.startswith(('+++', '---', '@@')):
            continue
            
        if line.startswith('-'):
            content = line[1:].strip()
            if content:
                pending_removals.append(content)
                
        elif line.startswith('+'):
            content = line[1:].strip()
            if content:
                # If there's a pending removal, pair it up as a Modification
                if pending_removals:
                    old_text = pending_removals.pop(0)
                    diff_lines.append(f'Modified: "{old_text}" -> "{content}"')
                else:
                    diff_lines.append(f'Added: "{content}"')
                    
        else:
            # If we hit an unchanged context line (starts with space), 
            # flush any un-paired removals as actual deletions.
            for old_text in pending_removals:
                diff_lines.append(f'Removed: "{old_text}"')
            pending_removals = []

    # Flush any remaining unpaired removals at the very end
    for old_text in pending_removals:
        diff_lines.append(f'Removed: "{old_text}"')
        
    return "\n".join(diff_lines[:100])

# --- LLM Communication with Image-Matched Sampling Parameters ---
def query_llm(file_name, diff_text, server_process_ref):
    if not diff_text.strip():
        return "Minor formatting edits."

    # Configured according to sampling settings in image_971b64.png
    sys_prompt = "You are an activity log writer. Summarize the modifications on user's files into one short sentence. Reply with ONLY the sentence."
    user_payload = f"Changes:\n{diff_text}"

    payload = {
        "model": "google/gemma-4-E2B-it-qat-q4_0-gguf:Q4_0",
        "temperature": 1.0,      # Matched from image_971b64.png
        "top_k": 64,             # Matched from image_971b64.png
        "top_p": 0.95,           # Matched from image_971b64.png
        "min_p": 0.05,           # Matched from image_971b64.png
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_payload}
        ]
    }

    if DRY_RUN:
        print(f"\n--- [DRY-RUN] PROMPT FOR: {file_name} ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("----------------------------------------\n")
        return f"[DRY-RUN SUMMARY FOR {file_name}]"

    ensure_server_running(server_process_ref)

    try:
        req = urllib.request.Request(
            LLAMA_URL, 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            response_data = json.loads(res.read().decode("utf-8"))
            output = response_data["choices"][0]["message"]["content"].strip()
            # Strip outer quotes or bullet points if the LLM adds them
            output = re.sub(r'^["\'`\s\-]+|["\'`\s]+$', '', output)
            return output.split('\n')[0]
    except Exception as e:
        print(f"LLM Error on {file_name}: {e}")
        return "File updated."

# --- Lazy Server Readiness Poller ---
_SERVER_STARTED = False
def ensure_server_running(server_process_ref):
    global _SERVER_STARTED
    if _SERVER_STARTED or DRY_RUN:
        return

    print("Booting llama-server inference engine...")
    proc = subprocess.Popen(
        LLAMA_SERVER_CMD,
        cwd=LLAMA_SERVER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    server_process_ref.append(proc)

    print("Waiting for GGUF model weights to load into VRAM...")
    start_time = time.time()
    
    # Poll until llama-server returns a HTTP 200 with status: ok
    while time.time() - start_time < 120:
        try:
            req = urllib.request.Request("http://localhost:8080/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    data = json.loads(r.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        _SERVER_STARTED = True
                        print("Inference engine fully loaded and ready!")
                        return
        except Exception:
            pass
        time.sleep(2)

    print("Warning: Timed out waiting for llama-server. Proceeding cautiously...")
    _SERVER_STARTED = True

# --- Main Logic ---
def main():
    print(f"Scanning target directory: {TARGET_DIR}")
    if DRY_RUN:
        print(">>> DRY-RUN MODE ENABLED <<<")

    conn = init_db()
    cursor = conn.cursor()

    cursor.execute("SELECT path, content, hash FROM file_snapshots")
    db_rows = cursor.fetchall()
    old_files = {row[0]: {"content": row[1], "hash": row[2]} for row in db_rows}

    current_files = get_current_md_files(TARGET_DIR)

    if not old_files:
        print(" -> Initial run. Seeding database state...")
        if not DRY_RUN:
            for path, val in current_files.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO file_snapshots (path, content, hash) VALUES (?, ?, ?)",
                    (path, val["content"], val["hash"])
                )
            conn.commit()
        conn.close()
        print("Database initialized.")
        return

    potential_deletions = {p: v for p, v in old_files.items() if p not in current_files}
    potential_additions = {p: v for p, v in current_files.items() if p not in old_files}

    logs = []
    matched_deletions = set()
    matched_additions = set()
    server_process_container = []

    # 1. Moves/Renames (Exact Hash Match)
    for del_path, del_val in potential_deletions.items():
        for add_path, add_val in potential_additions.items():
            if add_path in matched_additions: continue
            if del_val["hash"] == add_val["hash"]:
                logs.append(f"File `{del_path}` was moved/renamed to `{add_path}`")
                matched_deletions.add(del_path)
                matched_additions.add(add_path)
                break

    # 2. Moves/Renames with Content Changes (Strict Fuzzy Match)
    for del_path, del_val in potential_deletions.items():
        if del_path in matched_deletions: continue
        del_base = os.path.basename(del_path)
        
        for add_path, add_val in potential_additions.items():
            if add_path in matched_additions: continue
            
            is_match = False
            # Same filename in a different folder
            if del_base == os.path.basename(add_path):
                is_match = True
            else:
                # STRICT content comparison: require at least 75% structural similarity
                # Uses full ratio(), avoiding false positives from quick estimates
                ratio = difflib.SequenceMatcher(None, del_val["content"], add_val["content"]).ratio()
                if ratio > 0.75:
                    is_match = True

            if is_match:
                diff_text = extract_clean_diff(del_val["content"], add_val["content"])
                summary = query_llm(add_path, diff_text, server_process_container)
                logs.append(f"File `{del_path}` was moved/renamed to `{add_path}`: {summary}")
                matched_deletions.add(del_path)
                matched_additions.add(add_path)
                break

    # 3. Pure Deletions
    for del_path in potential_deletions:
        if del_path not in matched_deletions:
            logs.append(f"File `{del_path}` was deleted")

    # 4. Pure Creations
    for add_path, add_val in potential_additions.items():
        if add_path not in matched_additions:
            diff_text = extract_clean_diff("", add_val["content"])
            summary = query_llm(add_path, diff_text, server_process_container)
            logs.append(f"File `{add_path}`: {summary}")

    # 5. Modified Files
    for path, curr_val in current_files.items():
        if path in old_files and old_files[path]["hash"] != curr_val["hash"]:
            old_val = old_files[path]
            diff_text = extract_clean_diff(old_val["content"], curr_val["content"])
            summary = query_llm(path, diff_text, server_process_container)
            logs.append(f"File `{path}`: {summary}")

    # Write Output
    if logs:
        today_str = datetime.now().strftime("%Y/%m/%d")
        if DRY_RUN:
            print(f"\n--- [DRY-RUN] GENERATED LOG ENTRIES ({today_str}) ---")
            for entry in logs:
                print(entry)
            print("----------------------------------------------------\n")
        else:
            with open(CHANGELOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n# {today_str}\n")
                for entry in logs:
                    f.write(f"{entry}\n")
            print(f"Changelog synchronized to {CHANGELOG_FILE}")

            # Update SQLite snapshot
            cursor.execute("DELETE FROM file_snapshots")
            for path, val in current_files.items():
                cursor.execute(
                    "INSERT INTO file_snapshots (path, content, hash) VALUES (?, ?, ?)",
                    (path, val["content"], val["hash"])
                )
            conn.commit()
    else:
        print("No modifications discovered today.")

    conn.close()

    # Graceful LLM Process Shutdown
    if server_process_container:
        proc = server_process_container[0]
        if proc and proc.poll() is None:
            print("Shutting down llama-server engine...")
            proc.send_signal(signal.SIGINT)
            proc.wait()
            print("Done.")

if __name__ == "__main__":
    main()
