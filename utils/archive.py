import shutil #ファイルやディレクトリのコピー、削除、移動，圧縮などを行うためのモジュール
from pathlib import Path

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
    else:
        raise ValueError(
            f"Unsupported archive type: {archive_type}"
            )