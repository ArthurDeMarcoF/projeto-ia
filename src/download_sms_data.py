"""Download the publicly available UCI SMS Spam Collection dataset."""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_sms(output_path: Path) -> Path:
    with urllib.request.urlopen(UCI_URL, timeout=45) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        matching = [item for item in archive.infolist()
                    if Path(item.filename).name == "SMSSpamCollection" and not item.is_dir()]
        if len(matching) != 1:
            raise ValueError("UCI archive does not contain the expected file")
        data = archive.read(matching[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/SMSSpamCollection"))
    args = parser.parse_args()
    print("Saved:", download_sms(args.output))


if __name__ == "__main__":
    main()
