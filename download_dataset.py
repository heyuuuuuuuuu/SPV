"""
从腾讯云 COS 下载数据集

用法:
  1. 设置环境变量 COS_SECRET_KEY（或直接修改 CONFIG）
  2. python download_dataset.py

配置:
  COS_SECRET_KEY  - 腾讯云 SecretKey（必填）
  DATASET_DIR     - 下载到本地哪个目录（默认: E:/dataset/char_spv_augmented）
  COS_PREFIX      - COS 上的路径前缀（默认: dataset/char_spv_augmented）
"""

import os
import sys
import time
import hashlib
import threading
from pathlib import Path
from queue import Queue

from qcloud_cos import CosConfig, CosS3Client

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

CONFIG = {
    "secret_id": os.environ.get("COS_SECRET_ID", ""),
    "secret_key": os.environ.get("COS_SECRET_KEY", ""),
    "region": "ap-guangzhou",
    "bucket": "12321321-1311063269",
    "scheme": "https",
}

# 本地数据集目录（下载目标）
DATASET_DIR = os.environ.get("DATASET_DIR", "E:/dataset/char_spv_augmented")

# COS 上前缀（云端路径）
COS_PREFIX = os.environ.get("COS_PREFIX", "dataset/char_spv_augmented")

# 下载线程数
MAX_WORKERS = 8

# 下载记录文件（断点续传）
PROGRESS_FILE = "download_progress.txt"


def get_client(config: dict) -> CosS3Client:
    """创建 COS 客户端"""
    cos_config = CosConfig(
        Region=config["region"],
        SecretId=config["secret_id"],
        SecretKey=config["secret_key"],
        Scheme=config["scheme"],
    )
    return CosS3Client(cos_config)


def get_file_md5(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件的 MD5"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_progress(progress_file: str) -> set:
    """加载已下载的文件列表"""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(progress_file: str, key: str):
    """保存已下载的文件记录"""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(key + "\n")


def list_all_objects(client: CosS3Client, bucket: str, prefix: str) -> list[dict]:
    """列出 COS 上 prefix 下的所有对象"""
    objects = []
    marker = ""
    while True:
        resp = client.list_objects(
            Bucket=bucket,
            Prefix=prefix,
            Marker=marker if marker else "",
            MaxKeys=1000,
        )
        contents = resp.get("Contents", [])
        objects.extend(contents)

        if resp.get("IsTruncated") == "true":
            marker = resp.get("NextMarker", "")
            if not marker and contents:
                marker = contents[-1].get("Key", "")
        else:
            break

    return objects


def download_file(
    client: CosS3Client,
    bucket: str,
    cos_key: str,
    local_path: str,
    retries: int = 3,
) -> bool:
    """从 COS 下载单个文件，带重试"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(retries):
        try:
            resp = client.get_object(Bucket=bucket, Key=cos_key)
            resp["Body"].get_stream_to_file(local_path)
            return True
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠ 重试 {attempt + 1}/{retries}（{wait}s 后）: {os.path.basename(local_path)}")
                time.sleep(wait)
            else:
                print(f"  ✗ 下载失败: {cos_key} → {local_path}\n    错误: {e}")
                return False
    return False


def download_worker(
    client: CosS3Client,
    bucket: str,
    base_dir: str,
    cos_prefix: str,
    task_queue: Queue,
    progress: set,
    lock: threading.Lock,
    stats: dict,
):
    """下载工作线程"""
    while not task_queue.empty():
        try:
            cos_key = task_queue.get_nowait()
        except:
            break

        # 计算本地路径
        rel_path = cos_key[len(cos_prefix):].lstrip("/")
        local_path = os.path.join(base_dir, rel_path)

        # 跳过已完成
        with lock:
            if cos_key in progress:
                stats["skipped"] += 1
                task_queue.task_done()
                continue

        # 下载
        success = download_file(client, bucket, cos_key, local_path)
        if success:
            with lock:
                progress.add(cos_key)
                save_progress(PROGRESS_FILE, cos_key)
                stats["downloaded"] += 1

        task_queue.task_done()


def main():
    # ── 检查密钥 ──
    missing = []
    if not CONFIG["secret_id"]:
        missing.append("COS_SECRET_ID")
    if not CONFIG["secret_key"]:
        missing.append("COS_SECRET_KEY")
    if missing:
        print(f"✗ 未设置环境变量: {', '.join(missing)}")
        print("  Windows:")
        print("    set COS_SECRET_ID=AKIDxxxx")
        print("    set COS_SECRET_KEY=你的SecretKey")
        print("  Linux:")
        print("    export COS_SECRET_ID=AKIDxxxx")
        print("    export COS_SECRET_KEY=你的SecretKey")
        sys.exit(1)

    # ── 显示配置 ──
    print("=" * 60)
    print("  腾讯云 COS 数据集下载工具")
    print("=" * 60)
    print(f"  Bucket:    {CONFIG['bucket']}")
    print(f"  Region:    {CONFIG['region']}")
    print(f"  本地目录:  {DATASET_DIR}")
    print(f"  COS 前缀:  {COS_PREFIX}")
    print(f"  线程数:    {MAX_WORKERS}")
    print("=" * 60)

    # ── 创建客户端 ──
    print("\n[1/4] 连接 COS...")
    client = get_client(CONFIG)

    # 验证连接
    try:
        client.head_bucket(Bucket=CONFIG["bucket"])
        print("  ✓ 连接成功")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        sys.exit(1)

    # ── 列出云端文件 ──
    print("\n[2/4] 扫描云端文件...")
    objects = list_all_objects(client, CONFIG["bucket"], CONFIG["COS_PREFIX"])
    if not objects:
        print(f"  ✗ COS 上未找到文件（prefix: {CONFIG['COS_PREFIX']}）")
        sys.exit(1)

    files = [obj["Key"] for obj in objects if not obj["Key"].endswith("/")]
    total_size = sum(obj.get("Size", 0) for obj in objects)
    print(f"  ✓ 共 {len(files)} 个文件, "
          f"{total_size / 1024 / 1024 / 1024:.2f} GB")

    # ── 加载进度（断点续传） ──
    progress = load_progress(PROGRESS_FILE)
    if progress:
        print(f"  📋 已完成 {len(progress)} 个文件（断点续传）")

    # ── 创建任务队列 ──
    task_queue: Queue = Queue()
    for cos_key in files:
        task_queue.put(cos_key)

    # ── 下载 ──
    print(f"\n[3/4] 开始下载（{MAX_WORKERS} 线程）...")
    print("-" * 60)

    stats = {"downloaded": 0, "skipped": 0}
    lock = threading.Lock()
    start_time = time.time()

    threads = []
    for i in range(MAX_WORKERS):
        t = threading.Thread(
            target=download_worker,
            args=(client, CONFIG["bucket"], DATASET_DIR, COS_PREFIX,
                  task_queue, progress, lock, stats),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # 进度显示
    total = len(files)
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        done = stats["downloaded"] + stats["skipped"]
        pct = done / total * 100 if total > 0 else 0
        elapsed = time.time() - start_time
        speed = done / elapsed if elapsed > 0 else 0
        print(f"  进度: {done}/{total} ({pct:.1f}%)  "
              f"速度: {speed:.1f} files/s  "
              f"已用: {elapsed:.0f}s")

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    print("-" * 60)

    # ── 完成 ──
    print(f"\n[4/4] ✓ 下载完成!")
    print(f"  下载: {stats['downloaded']} 个文件")
    print(f"  跳过: {stats['skipped']} 个文件（已完成）")
    print(f"  用时: {elapsed:.1f}s")
    print(f"  保存位置: {DATASET_DIR}")

    # 清理进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"  （已清理断点续传记录）")


if __name__ == "__main__":
    main()
