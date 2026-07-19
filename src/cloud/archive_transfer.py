"""腾讯云 COS 数据集归档传输。

将大量小文件打包为一个 tar.gz 对象上传，使用 SHA-256 清单校验；旧的零散对象
只能通过显式 cleanup 命令删除，避免在归档上传成功前误删数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from qcloud_cos import CosConfig, CosS3Client

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

DEFAULT_REGION = os.environ.get("COS_REGION", "ap-guangzhou")
DEFAULT_BUCKET = os.environ.get("COS_BUCKET", "12321321-1311063269")
DEFAULT_DATASET_NAME = os.environ.get("COS_DATASET_NAME", "spv-hanzi-augmented")
DEFAULT_DATASET_VERSION = os.environ.get("COS_DATASET_VERSION", "v1")
DEFAULT_LEGACY_PREFIX = os.environ.get(
    "COS_LEGACY_PREFIX", "dataset/char_spv_augmented"
).rstrip("/")
DEFAULT_DATASET_DIR = os.environ.get("DATASET_DIR", "data/dataset")
DEFAULT_ARCHIVE_KEY = os.environ.get(
    "COS_ARCHIVE_KEY",
    f"datasets/{DEFAULT_DATASET_NAME}/{DEFAULT_DATASET_VERSION}/"
    f"{DEFAULT_DATASET_NAME}-{DEFAULT_DATASET_VERSION}.tar.gz",
)


def get_client() -> CosS3Client:
    """根据环境变量创建并验证 COS 客户端。"""
    secret_id = os.environ.get("COS_SECRET_ID", "")
    secret_key = os.environ.get("COS_SECRET_KEY", "")
    if not secret_id or not secret_key:
        raise RuntimeError("请设置环境变量 COS_SECRET_ID 和 COS_SECRET_KEY")

    config = CosConfig(
        Region=DEFAULT_REGION,
        SecretId=secret_id,
        SecretKey=secret_key,
        Scheme="https",
    )
    client = CosS3Client(config)
    client.head_bucket(Bucket=DEFAULT_BUCKET)
    return client


# ═══════════════════════════════════════════════════════════
# 本地归档与校验
# ═══════════════════════════════════════════════════════════

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算文件 SHA-256，避免将大归档读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_dir: Path) -> tuple[int, int]:
    files = [path for path in dataset_dir.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def create_archive(dataset_dir: Path, archive_path: Path) -> dict[str, Any]:
    """创建 tar.gz；归档内保存数据集目录下的相对路径。"""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    if archive_path.resolve().is_relative_to(dataset_dir.resolve()):
        raise ValueError("归档文件不能放在数据集目录内部")

    file_count, source_size = scan_dataset(dataset_dir)
    if file_count == 0:
        raise ValueError(f"数据集目录为空: {dataset_dir}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(f".{archive_path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    print(
        f"创建归档: {file_count} 个文件，源数据 {source_size / 1024**3:.2f} GiB\n"
        f"  {dataset_dir} -> {archive_path}"
    )
    try:
        # compresslevel=1 对已压缩 PNG 更快；这类数据通常不会明显缩小。
        with tarfile.open(temp_path, "w:gz", compresslevel=1) as archive:
            for child in sorted(dataset_dir.iterdir(), key=lambda item: item.name):
                archive.add(child, arcname=child.name, recursive=True)
        temp_path.replace(archive_path)
    finally:
        temp_path.unlink(missing_ok=True)

    archive_size = archive_path.stat().st_size
    manifest = {
        "format": "tar.gz",
        "archive_name": archive_path.name,
        "archive_size": archive_size,
        "sha256": sha256_file(archive_path),
        "file_count": file_count,
        "source_size": source_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        f"归档完成: {archive_size / 1024**3:.2f} GiB，"
        f"SHA-256={manifest['sha256']}"
    )
    return manifest


def local_manifest(archive_path: Path) -> dict[str, Any]:
    return {
        "format": "tar.gz",
        "archive_name": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# COS 操作
# ═══════════════════════════════════════════════════════════

def upload_archive(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    archive_path = Path(args.archive_file or f"{dataset_dir}.tar.gz")

    if args.reuse_archive:
        if not archive_path.is_file():
            raise FileNotFoundError(f"待复用归档不存在: {archive_path}")
        print(f"复用已有归档: {archive_path}")
        manifest = local_manifest(archive_path)
    else:
        manifest = create_archive(dataset_dir, archive_path)

    client = get_client()
    print(f"上传归档到 cos://{DEFAULT_BUCKET}/{args.archive_key}")
    client.upload_file(
        Bucket=DEFAULT_BUCKET,
        Key=args.archive_key,
        LocalFilePath=str(archive_path),
        PartSize=args.part_size,
        MAXThread=args.workers,
        EnableMD5=True,
    )

    remote = client.head_object(Bucket=DEFAULT_BUCKET, Key=args.archive_key)
    remote_size = int(remote.get("Content-Length", -1))
    if remote_size != manifest["archive_size"]:
        raise RuntimeError(
            f"远端大小校验失败: local={manifest['archive_size']}, remote={remote_size}"
        )

    manifest.update({"bucket": DEFAULT_BUCKET, "key": args.archive_key})
    manifest_body = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        Bucket=DEFAULT_BUCKET,
        Key=f"{args.archive_key}.sha256.json",
        Body=manifest_body,
        ContentType="application/json",
    )
    print(f"上传及大小校验完成，清单: {args.archive_key}.sha256.json")

    if args.remove_local_archive:
        archive_path.unlink()
        print(f"已删除本地归档: {archive_path}")


def get_json_object(client: CosS3Client, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=DEFAULT_BUCKET, Key=key)
    return json.loads(response["Body"].get_raw_stream().read().decode("utf-8"))


def _validate_tar_members(archive: tarfile.TarFile) -> None:
    """拒绝路径穿越、绝对路径和链接。"""
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"归档包含不安全路径: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"归档包含不允许的链接: {member.name}")


def extract_archive(archive_path: Path, dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"目标目录非空: {dataset_dir}；确认覆盖时请添加 --overwrite"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _validate_tar_members(archive)
        archive.extractall(dataset_dir, filter="data")


def download_archive(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    archive_path = Path(args.archive_file or f"{dataset_dir}.tar.gz")
    client = get_client()

    manifest_key = f"{args.archive_key}.sha256.json"
    manifest = get_json_object(client, manifest_key)
    expected_size = int(manifest["archive_size"])
    expected_sha256 = str(manifest["sha256"])

    valid_local = (
        archive_path.is_file()
        and archive_path.stat().st_size == expected_size
        and sha256_file(archive_path) == expected_sha256
    )
    if valid_local:
        print(f"本地归档校验通过，跳过下载: {archive_path}")
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"下载 cos://{DEFAULT_BUCKET}/{args.archive_key}\n"
            f"  -> {archive_path} ({expected_size / 1024**3:.2f} GiB)"
        )
        client.download_file(
            Bucket=DEFAULT_BUCKET,
            Key=args.archive_key,
            DestFilePath=str(archive_path),
            PartSize=args.part_size,
            MAXThread=args.workers,
            EnableCRC=True,
            DumpRecordDir=str(archive_path.parent),
        )
        actual_sha256 = sha256_file(archive_path)
        if archive_path.stat().st_size != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError("下载归档的大小或 SHA-256 校验失败")
        print("下载校验通过")

    print(f"解包到: {dataset_dir}")
    extract_archive(archive_path, dataset_dir, args.overwrite)
    print("解包完成")
    if args.remove_local_archive:
        archive_path.unlink()
        print(f"已删除本地归档: {archive_path}")


def list_all_objects(client: CosS3Client, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    marker = ""
    while True:
        response = client.list_objects(
            Bucket=DEFAULT_BUCKET, Prefix=prefix, Marker=marker, MaxKeys=1000
        )
        contents = response.get("Contents", [])
        objects.extend(contents)
        if response.get("IsTruncated") != "true":
            return objects
        marker = response.get("NextMarker") or contents[-1]["Key"]


def cleanup_legacy(args: argparse.Namespace) -> None:
    """删除旧前缀下的零散对象，但不触碰新归档。"""
    if not args.yes:
        raise RuntimeError("危险操作未执行；请检查参数后添加 --yes")

    legacy_prefix = args.legacy_prefix.rstrip("/") + "/"
    if args.archive_key.startswith(legacy_prefix):
        raise ValueError("归档位于待删除前缀内，拒绝清理")

    client = get_client()
    # 清理前必须确认新归档及校验清单都存在。
    client.head_object(Bucket=DEFAULT_BUCKET, Key=args.archive_key)
    client.head_object(
        Bucket=DEFAULT_BUCKET, Key=f"{args.archive_key}.sha256.json"
    )
    objects = list_all_objects(client, legacy_prefix)
    keys = [item["Key"] for item in objects]
    print(f"将删除旧前缀 {legacy_prefix} 下 {len(keys)} 个零散对象")
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        response = client.delete_objects(
            Bucket=DEFAULT_BUCKET,
            Delete={"Object": [{"Key": key} for key in batch], "Quiet": "true"},
        )
        errors = response.get("Error", [])
        if errors:
            raise RuntimeError(f"部分对象删除失败: {errors[:3]}")
        print(f"  已删除 {min(start + len(batch), len(keys))}/{len(keys)}")
    print("旧零散对象清理完成")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="腾讯云 COS 数据集归档传输")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
        subparser.add_argument("--archive-file", help="本地 tar.gz 路径")
        subparser.add_argument("--archive-key", default=DEFAULT_ARCHIVE_KEY)
        subparser.add_argument("--part-size", type=int, default=16, help="分片大小 MiB")
        subparser.add_argument("--workers", type=int, default=8)
        subparser.add_argument("--remove-local-archive", action="store_true")

    upload = subparsers.add_parser("upload", help="打包并上传")
    add_common(upload)
    upload.add_argument("--reuse-archive", action="store_true", help="复用已有归档")
    upload.set_defaults(func=upload_archive)

    download = subparsers.add_parser("download", help="下载、校验并解包")
    add_common(download)
    download.add_argument("--overwrite", action="store_true", help="允许写入非空目录")
    download.set_defaults(func=download_archive)

    cleanup = subparsers.add_parser("cleanup", help="删除旧的零散对象")
    cleanup.add_argument("--legacy-prefix", default=DEFAULT_LEGACY_PREFIX)
    cleanup.add_argument("--archive-key", default=DEFAULT_ARCHIVE_KEY)
    cleanup.add_argument("--yes", action="store_true", help="确认执行删除")
    cleanup.set_defaults(func=cleanup_legacy)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
