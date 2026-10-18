"""
Test common utilities. Example run:

python -m pytest tests/test_common.py -v
"""

import os
import pytest
from nanochat.common import download_file_with_lock


def test_download_retries_after_failed_postprocess(tmp_path, monkeypatch):
    # e.g. Ctrl-C while eval_bundle.zip is being unzipped: the next run must redo the
    # postprocess, not find the file already in place and skip it forever
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    source = tmp_path / "bundle.zip"
    source.write_bytes(b"payload")
    url = source.as_uri()

    def failing_postprocess(file_path):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        download_file_with_lock(url, "bundle.zip", postprocess_fn=failing_postprocess)
    assert not os.path.exists(tmp_path / "base" / "bundle.zip")

    postprocessed = []
    file_path = download_file_with_lock(url, "bundle.zip", postprocess_fn=postprocessed.append)
    assert len(postprocessed) == 1
    with open(file_path, "rb") as f:
        assert f.read() == b"payload"
