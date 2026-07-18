"""
上传/下载数据集到腾讯云 COS（纯标准库，无需安装 cos-python-sdk-v5）

用法:
  上传:
    set COS_SECRET_KEY=你的SecretKey
    python cos_transfer.py upload

  下载:
    set COS_SECRET_KEY=你的SecretKey
    python cos_transfer.py download

参考: https://github.com/tencentyun/cos-python-sdk-v5/blob/master/qcloud_cos/cos_auth.py
"""

import os
import sys
import hmac
import time
import json
import queue
import hashlib
import threading
import http.client
from pathlib import Path
from urllib.parse import quote

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

SECRET_ID  = os.environ.get("COS_SECRET_ID", "")
SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
APPID      = "1311063269"
BUCKET     = "12321321"
REGION     = "ap-guangzhou"
HOST       = f"{BUCKET}-{APPID}.cos.{REGION}.myqcloud.com"

DATASET_DIR  = os.environ.get("DATASET_DIR", "E:/dataset/char_spv_augmented")
COS_PREFIX   = os.environ.get("COS_PREFIX", "dataset/char_spv_augmented")
MAX_WORKERS  = 8
PROGRESS_FILE_UPLOAD   = "cos_upload_progress.txt"
PROGRESS_FILE_DOWNLOAD = "cos_download_progress.txt"


# ═══════════════════════════════════════════════════════════
# COS 签名（参考 cos_auth.py）
# ═══════════════════════════════════════════════════════════

def cos_sign(method: str, path: str, headers: dict = None,
             params: dict = None, expire: int = 3600) -> str:
    """
    生成 COS Authorization 签名头
    """
    if headers is None:
        headers = {}
    if params is None:
        params = {}

    # 过滤参与签名的头部
    valid = [
        "cache-control", "content-disposition", "content-encoding",
        "content-type", "content-md5", "content-length", "expect",
        "expires", "host", "if-match", "if-modified-since",
        "if-none-match", "if-unmodified-since", "origin", "range",
        "transfer-encoding", "pic-operations",
    ]
    sign_headers = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in valid or kl.startswith("x-cos-") or kl.startswith("x-ci-"):
            sign_headers[k] = v

    # URL 编码 key 和 value
    enc_headers = {}
    for k, v in sign_headers.items():
        enc_headers[quote(k.lower(), '-_.~')] = quote(str(v), '-_.~')

    enc_params = {}
    for k, v in params.items():
        enc_params[quote(k.lower(), '-_.~')] = quote(str(v), '-_.~')

    # 格式化字符串
    param_str = '&'.join(f"{k}={v}" for k, v in sorted(enc_params.items()))
    header_str = '&'.join(f"{k}={v}" for k, v in sorted(enc_headers.items()))

    format_str = f"{method.lower()}\n{path}\n{param_str}\n{header_str}\n"

    # 签名时间
    start_time = int(time.time())
    sign_time = f"{start_time - 60};{start_time + expire}"

    sha1 = hashlib.sha1(format_str.encode()).hexdigest()
    str_to_sign = f"sha1\n{sign_time}\n{sha1}\n"

    sign_key = hmac.new(
        SECRET_KEY.encode(), sign_time.encode(), hashlib.sha1
    ).hexdigest()
    signature = hmac.new(
        sign_key.encode(), str_to_sign.encode(), hashlib.sha1
    ).hexdigest()

    header_list = ';'.join(sorted(enc_headers.keys()))
    param_list = ';'.join(sorted(enc_params.keys()))

    auth = (
        f"q-sign-algorithm=sha1"
        f"&q-ak={SECRET_ID}"
        f"&q-sign-time={sign_time}"
        f"&q-key-time={sign_time}"
        f"&q-header-list={header_list}"
        f"&q-url-param-list={param_list}"
        f"&q-signature={signature}"
    )

    return auth


# ═══════════════════════════════════════════════════════════
# COS HTTP 请求
# ═══════════════════════════════════════════════════════════

def cos_request(method: str, path: str, params: dict = None,
                body: bytes = None, headers: dict = None,
                retries: int = 3) -> tuple[int, bytes]:
    """
    发送 COS HTTP 请求，返回 (status_code, response_body)
    
    path: 原始未编码路径（用于签名）
    """
    if headers is None:
        headers = {}

    # 确保 path 以 / 开头
    if not path.startswith('/'):
        path = '/' + path

    # HTTP 请求使用 URL 编码后的路径
    encoded_path = quote(path, safe='/-_.~')

    # 构建完整 URI（encoded path + query string）
    if params:
        from urllib.parse import urlencode
        qs = urlencode(sorted(params.items()))
        full_path = encoded_path + '?' + qs
    else:
        full_path = encoded_path
        params = {}

    # 设置默认 headers
    headers.setdefault("Host", HOST)

    if body is not None:
        headers.setdefault("Content-Length", str(len(body)))

    # 生成签名（使用原始未编码路径）
    auth = cos_sign(method, path, headers, params)
    headers["Authorization"] = auth

    for attempt in range(retries):
        try:
            conn = http.client.HTTPSConnection(HOST, timeout=120)
            conn.request(method, full_path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            conn.close()
            return resp.status, resp_body
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

    return -1, b""


def cos_put_object(cos_key: str, local_path: str,
                   content_type: str = "application/octet-stream") -> bool:
    """上传单个文件到 COS"""
    path = '/' + cos_key if not cos_key.startswith('/') else cos_key

    with open(local_path, "rb") as f:
        body = f.read()

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }

    status, resp = cos_request("PUT", path, body=body, headers=headers)

    if status == 200:
        return True
    else:
        print(f"    HTTP {status}: {resp.decode(errors='replace')[:200]}")
        return False


def cos_get_object(cos_key: str, local_path: str) -> bool:
    """从 COS 下载单个文件"""
    path = '/' + cos_key if not cos_key.startswith('/') else cos_key

    status, body = cos_request("GET", path)

    if status == 200:
        dirname = os.path.dirname(local_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(body)
        return True
    else:
        print(f"    HTTP {status}: {body.decode(errors='replace')[:200]}")
        return False


def cos_list_objects(prefix: str, marker: str = "",
                     max_keys: int = 1000) -> list[dict]:
    """列出 COS 上的对象"""
    objects = []

    while True:
        params = {"prefix": prefix, "marker": marker, "max-keys": str(max_keys)}

        status, body = cos_request("GET", "/", params=params)

        if status != 200:
            print(f"  ListObjects HTTP {status}: {body.decode(errors='replace')[:200]}")
            break

        # 手动解析 XML（避免依赖 xml 库版本问题）
        xml_str = body.decode("utf-8")
        contents = _parse_list_objects_xml(xml_str)
        objects.extend(contents)

        is_truncated = _xml_value(xml_str, "IsTruncated")
        if is_truncated == "true" and contents:
            marker = contents[-1].get("Key", "")
        else:
            break

    return objects


# ═══════════════════════════════════════════════════════════
# 简单 XML 解析
# ═══════════════════════════════════════════════════════════

def _xml_value(xml: str, tag: str) -> str:
    """提取 <tag>value</tag>"""
    import re
    m = re.search(f"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
    return m.group(1) if m else ""


def _parse_list_objects_xml(xml: str) -> list[dict]:
    """解析 ListBucketResult XML"""
    import re
    objects = []
    for match in re.finditer(r"<Contents>(.*?)</Contents>", xml, re.DOTALL):
        block = match.group(1)
        obj = {
            "Key": _xml_value(block, "Key"),
            "Size": int(_xml_value(block, "Size") or 0),
        }
        objects.append(obj)
    return objects


# ═══════════════════════════════════════════════════════════
# 进度管理
# ═══════════════════════════════════════════════════════════

def load_progress(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(path: str, key: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(key + "\n")


# ═══════════════════════════════════════════════════════════
# 上传
# ═══════════════════════════════════════════════════════════

def upload_worker(task_queue: queue.Queue, progress: set,
                  lock: threading.Lock, stats: dict):
    while not task_queue.empty():
        try:
            local_path, cos_key = task_queue.get_nowait()
        except queue.Empty:
            break

        with lock:
            if cos_key in progress:
                stats["skipped"] += 1
                task_queue.task_done()
                continue

        success = cos_put_object(cos_key, local_path)
        if success:
            with lock:
                progress.add(cos_key)
                save_progress(PROGRESS_FILE_UPLOAD, cos_key)
                stats["uploaded"] += 1

        task_queue.task_done()


def cmd_upload():
    if not SECRET_ID or not SECRET_KEY:
        print("[X] 请设置环境变量 COS_SECRET_ID 和 COS_SECRET_KEY")
        sys.exit(1)

    print("=" * 60)
    print("  腾讯云 COS 数据集上传工具")
    print("=" * 60)
    print(f"  Host:      {HOST}")
    print(f"  本地目录:  {DATASET_DIR}")
    print(f"  COS 前缀:  {COS_PREFIX}")
    print("=" * 60)

    # 扫描文件
    print("\n[1/3] 扫描本地文件...")
    base_dir = Path(DATASET_DIR)
    if not base_dir.exists():
        print(f"[X] 目录不存在: {DATASET_DIR}")
        sys.exit(1)

    files = []
    for p in sorted(base_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base_dir).as_posix()
            cos_key = f"{COS_PREFIX}/{rel}"
            files.append((str(p), cos_key))

    total_size = sum(os.path.getsize(fp) for fp, _ in files)
    print(f"  [OK] {len(files)} 个文件, {total_size / 1024 / 1024 / 1024:.2f} GB")

    # 断点续传
    progress = load_progress(PROGRESS_FILE_UPLOAD)
    if progress:
        print(f"  [~] 已完成 {len(progress)} 个（断点续传）")

    # 上传
    print(f"\n[2/3] 开始上传（{MAX_WORKERS} 线程）...")
    print("-" * 60)

    task_queue = queue.Queue()
    for item in files:
        task_queue.put(item)

    stats = {"uploaded": 0, "skipped": 0}
    lock = threading.Lock()
    start = time.time()

    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=upload_worker,
                             args=(task_queue, progress, lock, stats), daemon=True)
        t.start()
        threads.append(t)

    total = len(files)
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        done = stats["uploaded"] + stats["skipped"]
        pct = done / total * 100 if total > 0 else 0
        elapsed = time.time() - start
        speed = done / elapsed if elapsed > 0 else 0
        print(f"  进度: {done}/{total} ({pct:.1f}%)  "
              f"速度: {speed:.1f} f/s  已用: {elapsed:.0f}s")

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print("-" * 60)
    print(f"\n[3/3] [OK] 完成! 上传: {stats['uploaded']}, 跳过: {stats['skipped']}, 用时: {elapsed:.0f}s")

    if os.path.exists(PROGRESS_FILE_UPLOAD):
        os.remove(PROGRESS_FILE_UPLOAD)

    print(f"\n  数据位于: https://{HOST}/{COS_PREFIX}/")


# ═══════════════════════════════════════════════════════════
# 下载
# ═══════════════════════════════════════════════════════════

def download_worker(task_queue: queue.Queue, progress: set,
                    lock: threading.Lock, stats: dict):
    while not task_queue.empty():
        try:
            cos_key = task_queue.get_nowait()
        except queue.Empty:
            break

        rel = cos_key[len(COS_PREFIX):].lstrip("/")
        local_path = os.path.join(DATASET_DIR, rel)

        with lock:
            if cos_key in progress:
                stats["skipped"] += 1
                task_queue.task_done()
                continue

        success = cos_get_object(cos_key, local_path)
        if success:
            with lock:
                progress.add(cos_key)
                save_progress(PROGRESS_FILE_DOWNLOAD, cos_key)
                stats["downloaded"] += 1

        task_queue.task_done()


def cmd_download():
    if not SECRET_ID or not SECRET_KEY:
        print("[X] 请设置环境变量 COS_SECRET_ID 和 COS_SECRET_KEY")
        sys.exit(1)

    print("=" * 60)
    print("  腾讯云 COS 数据集下载工具")
    print("=" * 60)
    print(f"  Host:      {HOST}")
    print(f"  本地目录:  {DATASET_DIR}")
    print(f"  COS 前缀:  {COS_PREFIX}")
    print("=" * 60)

    # 列出云端文件
    print("\n[1/3] 扫描云端文件...")
    objects = cos_list_objects(COS_PREFIX)
    files = [obj["Key"] for obj in objects if not obj["Key"].endswith("/")]
    total_size = sum(obj.get("Size", 0) for obj in objects)

    if not files:
        print(f"  [X] 未找到文件 (prefix={COS_PREFIX})")
        sys.exit(1)

    print(f"  [OK] {len(files)} 个文件, {total_size / 1024 / 1024 / 1024:.2f} GB")

    # 断点续传
    progress = load_progress(PROGRESS_FILE_DOWNLOAD)
    if progress:
        print(f"  [~] 已完成 {len(progress)} 个（断点续传）")

    # 下载
    print(f"\n[2/3] 开始下载（{MAX_WORKERS} 线程）...")
    print("-" * 60)

    task_queue = queue.Queue()
    for key in files:
        task_queue.put(key)

    stats = {"downloaded": 0, "skipped": 0}
    lock = threading.Lock()
    start = time.time()

    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=download_worker,
                             args=(task_queue, progress, lock, stats), daemon=True)
        t.start()
        threads.append(t)

    total = len(files)
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        done = stats["downloaded"] + stats["skipped"]
        pct = done / total * 100 if total > 0 else 0
        elapsed = time.time() - start
        speed = done / elapsed if elapsed > 0 else 0
        print(f"  进度: {done}/{total} ({pct:.1f}%)  "
              f"速度: {speed:.1f} f/s  已用: {elapsed:.0f}s")

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print("-" * 60)
    print(f"\n[3/3] [OK] 完成! 下载: {stats['downloaded']}, 跳过: {stats['skipped']}, 用时: {elapsed:.0f}s")

    if os.path.exists(PROGRESS_FILE_DOWNLOAD):
        os.remove(PROGRESS_FILE_DOWNLOAD)

    print(f"\n  本地目录: {DATASET_DIR}")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  export COS_SECRET_ID=AKIDxxx")
        print("  export COS_SECRET_KEY=xxx")
        print("  python cos_transfer.py upload     # 上传数据集到 COS")
        print("  python cos_transfer.py download   # 从 COS 下载数据集")
        print()
        print("需先设置环境变量: COS_SECRET_ID, COS_SECRET_KEY")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "upload":
        cmd_upload()
    elif cmd == "download":
        cmd_download()
    else:
        print(f"未知命令: {cmd}")
        print("请使用: upload 或 download")
        sys.exit(1)
