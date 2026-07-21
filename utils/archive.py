"""
archive.py

Archive utilities for prediction results.

This module provides helper functions to:
- compress prediction directories into ZIP archives
- create TAR archives
- create TAR.ZST archives for efficient storage and transfer
"""

import shutil #ファイルやディレクトリのコピー、削除、移動，圧縮などを行うためのモジュール
from pathlib import Path
import subprocess
import tarfile

def archive_directory(
    directory,
    archive_type="zip",
):
    """指定フォルダをZIP、TAR、またはTAR.ZST形式で圧縮する。"""
    directory = Path(directory)
    
    archive_dir = Path("archives")
    archive_dir.mkdir(exist_ok=True)

    archive_path = archive_dir / directory.name
    
    if archive_type == "zip":
        shutil.make_archive(
            base_name=str(archive_path),
            format="zip",
            root_dir=str(directory.parent), # どこを基準に圧縮するか
            base_dir=directory.name, # その中のどのフォルダを圧縮するか
        )
        print(f"Saved: {archive_path}.zip")
        
    elif archive_type == "tar":
        shutil.make_archive(
            base_name=str(archive_path),
            format="tar",
            root_dir=str(directory.parent),
            base_dir=directory.name,
        )

        print(f"Saved: {archive_path}.tar")
        
    elif archive_type == "tar.zst":
        if shutil.which("zstd") is None:
            raise RuntimeError(
                "zstd command not found. Please install zstd to use tar.zst compression."
            )
            
        # まず　.tar を作る
        tar_path = shutil.make_archive(
            base_name=str(archive_path),
            format="tar",
            root_dir=str(directory.parent),
            base_dir=directory.name,
        )
        
        zst_path = f"{tar_path}.zst"
        
        # その後　.tar.zst に変換する
        subprocess.run(
            ["zstd", "-f", tar_path],
            check=True,
        )
            
        # .tar ファイルを削除する
        if Path(tar_path).exists():
            Path(tar_path).unlink()
        
        print(f"Saved: {zst_path}")
    
    else:
        raise ValueError(
            f"Unsupported archive type: {archive_type}"
            )


def _tar_zst_stem(archive_path):
    """ファイル名から複合拡張子 `.tar.zst` を取り除く。"""
    name = Path(archive_path).name
    suffix = ".tar.zst"
    if not name.endswith(suffix):
        raise ValueError(f"Expected a .tar.zst file, got: {name}")
    return name[:-len(suffix)]


def _safe_extract_tar(tar, output_dir):
    """危険なパスやリンクを拒否し、安全なファイルだけを展開する。"""
    output_dir = Path(output_dir).resolve()
    for member in tar:
        destination = (output_dir / member.name).resolve()
        if output_dir != destination and output_dir not in destination.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in archive: {member.name}")
        if not (member.isdir() or member.isfile()):
            continue
        tar.extract(member, path=output_dir)


def extract_tar_zst(archive_path, output_dir=None):
    """TAR.ZSTを展開し、展開先フォルダのPathを返す。"""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if output_dir is None:
        output_dir = archive_path.parent / f"{_tar_zst_stem(archive_path)}_extracted"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import zstandard as zstd
    except ImportError:
        zstd = None

    if zstd is not None:
        with archive_path.open("rb") as compressed_file:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(compressed_file) as tar_stream:
                with tarfile.open(fileobj=tar_stream, mode="r|") as tar:
                    _safe_extract_tar(tar, output_dir)
    else:
        if shutil.which("zstd") is None:
            raise RuntimeError(
                "Either the 'zstandard' Python package or the 'zstd' command "
                "is required to extract .tar.zst archives."
            )
        temporary_tar = output_dir / f"{_tar_zst_stem(archive_path)}.tar"
        subprocess.run(
            ["zstd", "-d", "-f", str(archive_path), "-o", str(temporary_tar)],
            check=True,
        )
        try:
            with tarfile.open(temporary_tar, mode="r") as tar:
                _safe_extract_tar(tar, output_dir)
        finally:
            temporary_tar.unlink(missing_ok=True)

    print(f"Extracted: {archive_path} -> {output_dir}")
    return output_dir
