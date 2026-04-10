"""Утилиты для работы с S3-хранилищем через boto3."""
from __future__ import annotations

import io
import json
import os
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient


def _normalize_endpoint_url(url: str) -> str:
    """Нормализует endpoint URL для надёжного сравнения.

    Args:
        url: Исходный endpoint URL.

    Returns:
        Нормализованный URL в виде `scheme://host[:port]`.
    """
    parsed = urlparse(url)
    host: str = parsed.hostname or ""
    if host == "127.0.0.1":
        host = "localhost"
    scheme: str = parsed.scheme or "http"
    port: str = f":{parsed.port}" if parsed.port is not None else ""
    return f"{scheme}://{host}{port}"


def _load_mc_credentials(endpoint_url: str | None) -> tuple[str | None, str | None]:
    """Пытается найти access/secret key в конфиге `mc`.

    Поиск идёт по точному совпадению нормализованного endpoint URL с alias.url.
    Если endpoint_url не задан или файл `~/.mc/config.json` отсутствует, возвращает
    `(None, None)`.

    Args:
        endpoint_url: Endpoint S3-compatible хранилища.

    Returns:
        Кортеж `(access_key, secret_key)` или `(None, None)`, если совпадение не найдено.
    """
    if endpoint_url is None:
        return None, None

    mc_config_path: str = os.path.expanduser("~/.mc/config.json")
    if not os.path.exists(mc_config_path):
        return None, None

    try:
        with open(mc_config_path, "r", encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    target_url: str = _normalize_endpoint_url(endpoint_url)
    aliases: dict[str, Any] = config.get("aliases", {})
    alias_payload: Any
    for alias_payload in aliases.values():
        if not isinstance(alias_payload, dict):
            continue
        alias_url_raw: Any = alias_payload.get("url")
        if not isinstance(alias_url_raw, str):
            continue
        if _normalize_endpoint_url(alias_url_raw) != target_url:
            continue
        access_key: Any = alias_payload.get("accessKey")
        secret_key: Any = alias_payload.get("secretKey")
        if isinstance(access_key, str) and isinstance(secret_key, str):
            return access_key, secret_key

    return None, None


def parse_s3_url(s3_url: str) -> tuple[str, str, str | None]:
    """Разбирает S3 URL на имя бакета и базовый префикс.

    Поддерживаемые форматы:
    - ``s3://bucket/some/prefix``
    - ``s3://127.0.0.1:9001/bucket/some/prefix`` для локальных S3-compatible endpoint
    - ``https://bucket.s3.amazonaws.com/some/prefix``
    - ``https://bucket.s3.region.amazonaws.com/some/prefix``
    - ``http://127.0.0.1:9001/bucket/some/prefix`` для path-style endpoint

    Args:
        s3_url: URL S3-хранилища.

    Returns:
        Кортеж ``(bucket, key_prefix, endpoint_url)``, где ``key_prefix`` может
        быть пустой строкой, а ``endpoint_url`` нужен для S3-compatible storage.

    Raises:
        ValueError: Если схема URL не поддерживается.
    """
    parsed = urlparse(s3_url)
    bucket: str
    key_prefix: str
    endpoint_url: str | None = None

    if parsed.scheme == "s3":
        path_parts: list[str] = [part for part in parsed.path.split("/") if part]
        is_endpoint_style: bool = (
            ":" in parsed.netloc
            or parsed.netloc in {"localhost", "127.0.0.1"}
        )
        if is_endpoint_style:
            if not path_parts:
                raise ValueError(
                    "Expected bucket name in path for endpoint-style S3 URL: "
                    "s3://host:port/bucket/prefix"
                )
            bucket = path_parts[0]
            key_prefix = "/".join(path_parts[1:])
            endpoint_url = os.environ.get("S3_ENDPOINT_URL") or f"http://{parsed.netloc}"
        else:
            bucket = parsed.netloc
            key_prefix = parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https"):
        host_parts: list[str] = parsed.netloc.split(".")
        is_virtual_hosted_style: bool = len(host_parts) >= 3 and host_parts[1] == "s3"
        if is_virtual_hosted_style:
            bucket = host_parts[0]
            key_prefix = parsed.path.lstrip("/")
        else:
            path_parts = [part for part in parsed.path.split("/") if part]
            if not path_parts:
                raise ValueError(
                    "Expected bucket name in path for path-style S3 URL: "
                    "http(s)://host:port/bucket/prefix"
                )
            bucket = path_parts[0]
            key_prefix = "/".join(path_parts[1:])
            endpoint_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        raise ValueError(
            f"Unsupported S3 URL scheme: {parsed.scheme!r}. "
            "Expected s3://, http:// or https:// URL"
        )

    return bucket, key_prefix, endpoint_url


def make_s3_client(endpoint_url: str | None = None) -> BaseClient:
    """Создаёт клиент boto3 S3.

    Учётные данные берутся из переменных окружения
    (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`,
    `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`) или из `~/.aws/credentials`.
    Для локального MinIO дополнительно поддерживается чтение `~/.mc/config.json`,
    если endpoint совпадает с alias.url.

    Args:
        endpoint_url: Явный endpoint URL для S3-compatible storage.

    Returns:
        Экземпляр boto3 S3 client.
    """
    resolved_endpoint_url: str | None = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
    access_key_id: str | None = (
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("S3_ACCESS_KEY_ID")
    )
    secret_access_key: str | None = (
        os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("S3_SECRET_ACCESS_KEY")
    )

    if access_key_id is None or secret_access_key is None:
        mc_access_key: str | None
        mc_secret_key: str | None
        mc_access_key, mc_secret_key = _load_mc_credentials(resolved_endpoint_url)
        access_key_id = access_key_id or mc_access_key
        secret_access_key = secret_access_key or mc_secret_key

    return boto3.client(
        "s3",
        endpoint_url=resolved_endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )


def list_video_keys(bucket: str, prefix: str, endpoint_url: str | None = None) -> list[str]:
    """Возвращает список ключей объектов в S3-бакете с заданным префиксом.

    Перебирает все страницы результатов paginator'а (list_objects_v2).
    Папки (ключи, заканчивающиеся на ``/``) пропускаются.

    Args:
        bucket: Имя S3-бакета.
        prefix: Префикс для фильтрации объектов.
        endpoint_url: Необязательный S3 endpoint для S3-compatible storage.

    Returns:
        Список ключей объектов.

    Raises:
        botocore.exceptions.BotoCoreError: При ошибках обращения к S3.
        botocore.exceptions.ClientError: При ошибках ответа S3.
    """
    client: BaseClient = make_s3_client(endpoint_url=endpoint_url)
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    page: dict[str, Any]
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        obj: dict[str, Any]
        for obj in page.get("Contents", []):
            key: str = str(obj["Key"])
            if not key.endswith("/"):  # skip directory markers
                keys.append(key)
    return keys


def download_to_memory(bucket: str, key: str, endpoint_url: str | None = None) -> bytes:
    """Скачивает объект из S3 целиком в память и возвращает байты.

    Не создаёт временных файлов на диске.

    Args:
        bucket: Имя S3-бакета.
        key: Ключ объекта в бакете.
        endpoint_url: Необязательный S3 endpoint для S3-compatible storage.

    Returns:
        Содержимое объекта в виде байтов.

    Raises:
        botocore.exceptions.BotoCoreError: При ошибках обращения к S3.
        botocore.exceptions.ClientError: При ошибках ответа S3.
    """
    client: BaseClient = make_s3_client(endpoint_url=endpoint_url)
    buf: io.BytesIO = io.BytesIO()
    client.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.read()


def upload_json_bytes(
    bucket: str,
    key: str,
    payload: bytes,
    endpoint_url: str | None = None,
) -> None:
    """Загружает JSON-документ в S3 как объект по указанному ключу.

    Args:
        bucket: Имя S3-бакета.
        key: Ключ объекта назначения.
        payload: UTF-8 байты JSON-документа.
        endpoint_url: Необязательный S3 endpoint для S3-compatible storage.

    Raises:
        botocore.exceptions.BotoCoreError: При ошибках обращения к S3.
        botocore.exceptions.ClientError: При ошибках ответа S3.
    """
    client: BaseClient = make_s3_client(endpoint_url=endpoint_url)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json; charset=utf-8",
    )
