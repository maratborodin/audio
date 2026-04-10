"""Тесты для s3_client.py."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import s3_client


class TestParseS3Url:
    """Проверки разбора S3 URL в bucket, prefix и endpoint."""

    def test_parse_standard_s3_url(self) -> None:
        """Проверяет разбор обычного `s3://bucket/prefix`.

        Returns:
            `None`.
        """
        bucket, prefix, endpoint = s3_client.parse_s3_url("s3://bucket/videos")
        assert bucket == "bucket"
        assert prefix == "videos"
        assert endpoint is None

    def test_parse_endpoint_style_s3_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет разбор endpoint-style URL для MinIO/S3-compatible.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
        bucket, prefix, endpoint = s3_client.parse_s3_url(
            "s3://127.0.0.1:9000/test-bucket/input_video"
        )
        assert bucket == "test-bucket"
        assert prefix == "input_video"
        assert endpoint == "http://127.0.0.1:9000"

    def test_parse_http_path_style_url(self) -> None:
        """Проверяет разбор path-style HTTP URL.

        Returns:
            `None`.
        """
        bucket, prefix, endpoint = s3_client.parse_s3_url(
            "http://localhost:9000/test-bucket/input_video"
        )
        assert bucket == "test-bucket"
        assert prefix == "input_video"
        assert endpoint == "http://localhost:9000"

    def test_parse_virtual_hosted_style_url(self) -> None:
        """Проверяет разбор virtual-hosted-style AWS URL.

        Returns:
            `None`.
        """
        bucket, prefix, endpoint = s3_client.parse_s3_url(
            "https://bucket.s3.amazonaws.com/input_video"
        )
        assert bucket == "bucket"
        assert prefix == "input_video"
        assert endpoint is None

    def test_parse_invalid_scheme_raises(self) -> None:
        """Проверяет ошибку для неподдерживаемой схемы URL.

        Returns:
            `None`.
        """
        with pytest.raises(ValueError, match="Unsupported S3 URL scheme"):
            s3_client.parse_s3_url("ftp://bucket/path")


class TestCredentialsAndClient:
    """Проверки поиска кредов и создания boto3-клиента."""

    def test_normalize_endpoint_url(self) -> None:
        """Проверяет нормализацию `127.0.0.1` к `localhost`.

        Returns:
            `None`.
        """
        assert (
            s3_client._normalize_endpoint_url("http://127.0.0.1:9000")
            == "http://localhost:9000"
        )

    def test_load_mc_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Проверяет чтение access/secret key из `~/.mc/config.json`.

        Args:
            monkeypatch: Фикстура monkeypatch.
            tmp_path: Временная директория pytest.

        Returns:
            `None`.
        """
        home: Path = tmp_path / "home"
        mc_dir: Path = home / ".mc"
        mc_dir.mkdir(parents=True)
        (mc_dir / "config.json").write_text(
            json.dumps(
                {
                    "aliases": {
                        "local": {
                            "url": "http://localhost:9000",
                            "accessKey": "minioadmin",
                            "secretKey": "minioadmin",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(home))

        access_key: str | None
        secret_key: str | None
        access_key, secret_key = s3_client._load_mc_credentials(
            "http://127.0.0.1:9000"
        )

        assert access_key == "minioadmin"
        assert secret_key == "minioadmin"

    def test_make_s3_client_uses_env_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет проброс endpoint и AWS credentials в boto3.client.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        captured: dict[str, object] = {}
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setattr(
            s3_client.boto3,
            "client",
            lambda service, **kwargs: captured.update(
                {"service": service, **kwargs}
            )
            or MagicMock(),
        )

        s3_client.make_s3_client("http://localhost:9000")

        assert captured["service"] == "s3"
        assert captured["endpoint_url"] == "http://localhost:9000"
        assert captured["aws_access_key_id"] == "key"


class TestS3Operations:
    """Проверки list/download/upload операций поверх boto3-клиента."""

    def test_list_video_keys_filters_directories(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет фильтрацию directory markers в выдаче S3.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        paginator: MagicMock = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "input/a.webm"}, {"Key": "input/folder/"}]}
        ]
        client: MagicMock = MagicMock()
        client.get_paginator.return_value = paginator
        monkeypatch.setattr(
            s3_client,
            "make_s3_client",
            lambda endpoint_url=None: client,
        )

        keys: list[str] = s3_client.list_video_keys(
            "bucket",
            "input",
            endpoint_url="http://localhost:9000",
        )

        assert keys == ["input/a.webm"]

    def test_download_to_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет скачивание объекта в `BytesIO`.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        client: MagicMock = MagicMock()

        def fake_download(bucket: str, key: str, buf: io.BytesIO) -> None:
            _ = bucket, key
            buf.write(b"payload")

        client.download_fileobj.side_effect = fake_download
        monkeypatch.setattr(
            s3_client,
            "make_s3_client",
            lambda endpoint_url=None: client,
        )

        payload: bytes = s3_client.download_to_memory("bucket", "key")

        assert payload == b"payload"

    def test_upload_json_bytes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет загрузку JSON как S3-объекта.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        client: MagicMock = MagicMock()
        monkeypatch.setattr(
            s3_client,
            "make_s3_client",
            lambda endpoint_url=None: client,
        )

        s3_client.upload_json_bytes(
            "bucket",
            "output_data/a.json",
            b"{}",
        )

        client.put_object.assert_called_once()
