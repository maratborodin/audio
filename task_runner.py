"""Фоновая обработка задачи: S3 → транскрипция, диаризация, выгрузка JSON.

Используется из ``main`` (пул потоков при локальном запуске) и из одноразовых
Docker-контейнеров (``python -m task_runner --task-id N``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from sqlalchemy.orm import Session

from db import SessionLocal, Task, TaskStatus, append_log, get_task, update_task
from process import load_whisper_pipeline, process_video_bytes
from s3_client import (
    download_to_memory,
    list_video_keys,
    parse_s3_url,
    upload_json_bytes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

DATA_PATH: str = os.environ.get("DATA_PATH", "./output_data")


def join_s3_prefixes(base_prefix: str, task_prefix: str) -> str:
    """Соединяет два S3-префикса без лишних или ведущих слэшей.

    Args:
        base_prefix: Базовый префикс из ``s3_url``.
        task_prefix: Дополнительный префикс из параметров задачи.

    Returns:
        Нормализованный S3-префикс без ведущего ``/``.
    """
    parts: list[str] = []
    normalized_base: str = base_prefix.strip("/")
    normalized_task: str = task_prefix.strip("/")
    if normalized_base:
        parts.append(normalized_base)
    if normalized_task:
        parts.append(normalized_task)
    return "/".join(parts)


def build_output_key(source_key: str) -> str:
    """Строит ключ назначения для результата в S3 под префиксом ``output_data/``.

    Args:
        source_key: Ключ исходного видеофайла в бакете.

    Returns:
        Ключ итогового JSON-файла в том же бакете.
    """
    file_name: str = os.path.splitext(os.path.basename(source_key))[0]
    return f"output_data/{file_name}.json"


def run_processing_task(task_id: int) -> None:
    """Выполняет полный цикл обработки видеофайлов из S3 для задачи ``task_id``.

    Загружает модель Whisper при первом использовании в этом процессе, обрабатывает
    все найденные видео, пишет логи и результат в SQLite.

    Args:
        task_id: Идентификатор задачи в базе данных.

    Raises:
        Exception: Не пробрасывает исключения наружу, но логирует и сохраняет
            их в статус задачи при ошибках инфраструктуры или обработки.
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    whisper_pipe: Any | None = None

    db: Session = SessionLocal()
    try:
        task: Task | None = get_task(db, task_id)
        if task is None:
            logger.error("Task %d not found in DB", task_id)
            return

        update_task(db, task, status=TaskStatus.RUNNING)
        append_log(db, task, "Task started")

        bucket: str
        base_prefix: str
        endpoint_url: str | None
        bucket, base_prefix, endpoint_url = parse_s3_url(task.s3_url)
        task_prefix: str = task.prefix if task.prefix else ""
        effective_prefix: str = join_s3_prefixes(base_prefix, task_prefix)
        append_log(
            db,
            task,
            f"S3 bucket={bucket!r} prefix={effective_prefix!r} endpoint={endpoint_url!r}",
        )

        keys: list[str] = list_video_keys(
            bucket,
            effective_prefix,
            endpoint_url=endpoint_url,
        )
        update_task(db, task, total=len(keys))
        append_log(db, task, f"Found {len(keys)} objects")

        if not keys:
            update_task(db, task, status=TaskStatus.DONE, result=[])
            append_log(db, task, "No files to process. Task done.")
            return

        append_log(db, task, "Loading Whisper model (may take minutes)…")
        whisper_pipe = load_whisper_pipeline()
        append_log(db, task, "Whisper model loaded")

        results: list[dict[str, Any]] = []
        key: str
        for i, key in enumerate(keys):
            file_name: str = os.path.splitext(os.path.basename(key))[0]
            update_task(db, task, current_file=key)
            append_log(db, task, f"[{i + 1}/{len(keys)}] Downloading {key}")

            try:
                video_bytes: bytes = download_to_memory(
                    bucket,
                    key,
                    endpoint_url=endpoint_url,
                )
                append_log(
                    db,
                    task,
                    f"[{i + 1}/{len(keys)}] Downloaded {len(video_bytes):,} bytes",
                )

                def _progress(msg: str, _task: Task = task, _db: Session = db) -> None:
                    """Проксирует сообщения прогресса из обработчика в лог задачи.

                    Args:
                        msg: Сообщение о текущем шаге обработки.
                        _task: ORM-объект задачи для обновления.
                        _db: Сессия базы данных.
                    """
                    append_log(_db, _task, msg)

                output_data: dict[str, Any] = process_video_bytes(
                    video_bytes=video_bytes,
                    file_name=file_name,
                    data_path=DATA_PATH,
                    pipe=whisper_pipe,
                    progress_callback=_progress,
                    save_output_locally=False,
                )
                output_key: str = build_output_key(key)
                upload_json_bytes(
                    bucket=bucket,
                    key=output_key,
                    payload=json.dumps(
                        output_data,
                        ensure_ascii=False,
                        indent=4,
                    ).encode("utf-8"),
                    endpoint_url=endpoint_url,
                )
                append_log(
                    db,
                    task,
                    f"[{i + 1}/{len(keys)}] Uploaded result to s3://{bucket}/{output_key}",
                )
                results.append({
                    "key": key,
                    "file_name": file_name,
                    "output_bucket": bucket,
                    "output_key": output_key,
                    "output_s3_url": f"s3://{bucket}/{output_key}",
                    "status": "ok",
                })
                append_log(db, task, f"[{i + 1}/{len(keys)}] Done: {file_name}")

            except Exception as exc:
                error_msg: str = f"{type(exc).__name__}: {exc}"
                append_log(db, task, f"[{i + 1}/{len(keys)}] ERROR {key}: {error_msg}")
                logger.exception("Task %d — error processing %s", task_id, key)
                results.append({
                    "key": key,
                    "file_name": file_name,
                    "status": "error",
                    "error": error_msg,
                })

            update_task(db, task, progress=i + 1)

        update_task(
            db,
            task,
            status=TaskStatus.DONE,
            result=results,
            current_file=None,
        )
        append_log(db, task, "Task completed successfully")

    except Exception as exc:
        logger.exception("Task %d failed with unhandled exception", task_id)
        task = get_task(db, task_id)
        if task is not None:
            update_task(
                db,
                task,
                status=TaskStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
            )
            append_log(db, task, f"FATAL ERROR: {exc}")
    finally:
        db.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Парсит аргументы CLI для одноразового воркера.

    Args:
        argv: Список аргументов или ``None`` для ``sys.argv[1:]``.

    Returns:
        Распознанные аргументы.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Обработка одной задачи (номер в БД) — для запуска в Docker-контейнере.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Идентификатор задачи в SQLite",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Точка входа CLI: ``python -m task_runner --task-id ID``."""
    args: argparse.Namespace = _parse_args(argv)
    run_processing_task(args.task_id)


if __name__ == "__main__":
    main()
