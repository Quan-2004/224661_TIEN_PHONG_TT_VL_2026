"""
main_pipeline.py – Khởi động toàn bộ hệ thống mOC-iSVM
=========================================================
Cách dùng:
    python main_pipeline.py

Script sẽ tự động:
  1. Kiểm tra & cài Python dependencies (backend/requirements.txt + mocsvm package)
  2. Kiểm tra & cài Node.js dependencies (frontend/package.json)
  3. Khởi động Backend  → http://localhost:8000  (FastAPI / uvicorn)
  4. Khởi động Frontend → http://localhost:5173  (Vite dev server)
  5. Mở trình duyệt tự động sau khi cả hai server sẵn sàng
  6. Nhấn Ctrl+C để dừng tất cả
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
import urllib.request
import urllib.error

# ─────────────────────────── Đường dẫn gốc ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

BACKEND_URL  = "http://localhost:8000/docs"
FRONTEND_URL = "http://localhost:5173"

# ──────────────────────────── Tiện ích ────────────────────────────────────

def log(msg: str, prefix: str = ">>") -> None:
    print(f"\n{prefix} {msg}", flush=True)


def wait_for_server(url: str, timeout: int = 60, label: str = "") -> bool:
    """Chờ server phản hồi HTTP 200, tối đa `timeout` giây, hiển thị tiến độ."""
    spinner = ["|", "/", "-", "\\"]
    deadline = time.time() + timeout
    elapsed = 0
    spin_idx = 0
    print(f"\n>> Đang chờ {label} khởi động ...", flush=True)
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            # Xoá dòng spinner, in thông báo OK
            print(f"\r  {label}: [{'█' * 20}] Sẵn sàng! ({elapsed}s)        ", flush=True)
            log(f"{label} sẵn sàng tại {url}", prefix="✔")
            return True
        except Exception:
            pass
        elapsed = int(time.time() - (deadline - timeout))
        filled = min(20, int(elapsed / timeout * 20))
        bar = "█" * filled + "░" * (20 - filled)
        spin = spinner[spin_idx % len(spinner)]
        print(f"\r  {label}: [{bar}] {spin} {elapsed}/{timeout}s ", end="", flush=True)
        spin_idx += 1
        time.sleep(1)
    print(f"\r  {label}: [{'░' * 20}] Timeout!                    ", flush=True)
    log(f"Timeout – {label} không khởi động được tại {url}", prefix="✘")
    return False


# ──────────────────────── Bước 1: Python deps ─────────────────────────────

def install_python_deps() -> None:
    req_file = os.path.join(BACKEND_DIR, "requirements.txt")
    log("Kiểm tra Python dependencies …")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
        cwd=ROOT,
    )

    # Cài mocsvm package ở chế độ editable (nếu chưa cài)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=ROOT,
    )
    log("Python dependencies OK.", prefix="✔")


# ──────────────────────── Bước 2: Node deps ───────────────────────────────

def install_node_deps() -> None:
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        log("node_modules đã tồn tại – bỏ qua npm install.", prefix="✔")
        return

    log("Cài Node.js dependencies (npm install) …")
    subprocess.check_call(["npm", "install"], cwd=FRONTEND_DIR, shell=True)
    log("Node.js dependencies OK.", prefix="✔")


# ──────────────────────── Bước 3+4: Khởi động server ─────────────────────

def start_backend() -> subprocess.Popen:
    log("Khởi động Backend (uvicorn) …")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--reload",
            "--port", "8000",
        ],
        cwd=ROOT,
    )


def start_frontend() -> subprocess.Popen:
    log("Khởi động Frontend (Vite) …")
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
    )


# ──────────────────────── Bước 5: Mở trình duyệt ─────────────────────────

def open_browser_when_ready() -> None:
    """Chờ frontend sẵn sàng rồi mở trình duyệt (chạy trên thread riêng)."""
    if wait_for_server(FRONTEND_URL, timeout=60, label="Frontend"):
        webbrowser.open(FRONTEND_URL)


# ──────────────────────────── Main ────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("   mOC-iSVM – Pipeline Khởi Động")
    print("=" * 60)

    # --- Cài dependencies ---
    try:
        install_python_deps()
    except subprocess.CalledProcessError as e:
        log(f"Lỗi cài Python deps: {e}", prefix="✘")
        sys.exit(1)

    try:
        install_node_deps()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log(
            f"Lỗi cài Node deps: {e}\n"
            "  → Hãy đảm bảo Node.js đã được cài (https://nodejs.org) và thử lại.",
            prefix="✘",
        )
        sys.exit(1)

    # --- Bước 1: Khởi động Backend trước, chờ sẵn sàng ---
    backend_proc = start_backend()

    backend_ok = wait_for_server(BACKEND_URL, timeout=90, label="Backend")
    if not backend_ok:
        log("Backend không khởi động được. Dừng chương trình.", prefix="✘")
        backend_proc.terminate()
        sys.exit(1)

    # --- Bước 2: Backend sẵn sàng → mới khởi động Frontend ---
    frontend_proc = start_frontend()

    # Mở trình duyệt trên thread nền (không block main loop)
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    print("\n" + "=" * 60)
    print("  Hệ thống đang chạy:")
    print(f"    Frontend : {FRONTEND_URL}")
    print(f"    Backend  : http://localhost:8000")
    print(f"    API Docs : {BACKEND_URL}")
    print("  Nhấn  Ctrl+C  để dừng tất cả.")
    print("=" * 60 + "\n")

    # --- Giữ script chạy, lắng nghe Ctrl+C ---
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        log("Đang tắt …")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        log("Đã dừng tất cả server. Tạm biệt!", prefix="✔")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print("\n" + "=" * 60, flush=True)
        print("  [LOI NGHIEM TRONG]", flush=True)
        traceback.print_exc()
        print("=" * 60, flush=True)
        sys.exit(1)
