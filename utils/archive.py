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

def archive_directory(
    directory,
    archive_type="zip",
):
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