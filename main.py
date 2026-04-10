"""FastAPI web-сервер для асинхронной обработки видеофайлов из S3.

Запуск:
    ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Интерактивная документация OpenAPI (Swagger UI): ``/api/docs``.
Статическая документация Sphinx (после ``sphinx-build``): ``/docs/``.

Обработку задач можно выполнять в фоновых потоках (локально) или одноразовыми
Docker-контейнерами (см. ``USE_DOCKER_WORKERS``, ``docker-compose.yml``).

Переменные окружения:
    DATA_PATH          — директория для сохранения JSON-результатов (по умолчанию ./output_data)
    DATABASE_URL       — SQLAlchemy URL для SQLite (по умолчанию sqlite:///./tasks.db)
    S3_ENDPOINT_URL    — эндпоинт для S3-совместимых хранилищ (MinIO, Yandex и т.д.)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION — учётные данные AWS
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from db import (
    SessionLocal,
    Task,
    TaskStatus,
    append_log,
    create_task,
    get_task,
    list_tasks,
    update_task,
)
from docker_runner import spawn_worker_container, use_docker_workers
from task_runner import (
    build_output_key as _build_output_key,
    join_s3_prefixes as _join_s3_prefixes,
    run_processing_task as _run_processing_task,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

DATA_PATH: str = os.environ.get("DATA_PATH", "./output_data")

# Каталог со сборкой Sphinx: docs/build/html (раздаётся с URL ``/docs/``)
_DOCS_HTML_DIR: Path = Path(__file__).resolve().parent / "docs" / "build" / "html"

# Пул потоков для локального режима (USE_DOCKER_WORKERS=0): одна задача за раз.
_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)


def _submit_background_task(task_id: int) -> None:
    """Ставит обработку задачи: Docker-контейнер или фоновый поток.

    Args:
        task_id: Идентификатор задачи в БД.
    """
    if use_docker_workers():
        try:
            spawn_worker_container(task_id)
        except RuntimeError as exc:
            logger.exception("Не удалось запустить Docker-воркер для задачи %d", task_id)
            db: Session = SessionLocal()
            try:
                task_err: Task | None = get_task(db, task_id)
                if task_err is not None:
                    update_task(db, task_err, status=TaskStatus.ERROR, error=str(exc))
                    append_log(db, task_err, f"Ошибка запуска контейнера: {exc}")
            finally:
                db.close()
        return
    _executor.submit(_run_processing_task, task_id)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управляет жизненным циклом FastAPI-приложения.

    При старте создаёт локальную директорию ``DATA_PATH``, а при остановке
    завершает пул потоков (если используется локальный режим без Docker-воркеров).

    Args:
        app: Экземпляр FastAPI-приложения.

    Yields:
        Управление жизненным циклом приложения.
    """
    _ = app
    os.makedirs(DATA_PATH, exist_ok=True)
    logger.info("DATA_PATH: %s", DATA_PATH)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Audio Processing API",
    description="Асинхронная обработка видеофайлов из S3: транскрипция, диаризация, оценка голоса.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


def _mount_sphinx_html(application: FastAPI) -> None:
    """Подключает статическую раздачу HTML, собранного Sphinx (``docs/build/html``).

    Содержимое доступно по префиксу URL ``/docs/``. Если каталог отсутствует,
    в лог пишется предупреждение, приложение продолжает работу без раздачи.

    Args:
        application: Экземпляр FastAPI-приложения.
    """
    if not _DOCS_HTML_DIR.is_dir():
        logger.warning(
            "Sphinx HTML не найден (%s). Выполните: sphinx-build -b html docs/source docs/build/html",
            _DOCS_HTML_DIR,
        )
        return
    application.mount(
        "/docs",
        StaticFiles(directory=str(_DOCS_HTML_DIR), html=True),
        name="sphinx_docs",
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/process",
    summary="Запустить обработку видеофайлов из S3",
    response_description="Номер созданной задачи и её начальный статус",
)
async def start_process(
    s3_url: str = Query(
        ...,
        description="URL S3-хранилища, например s3://my-bucket/videos или https://bucket.s3.amazonaws.com/videos",
    ),
    prefix: str = Query(
        default="",
        description="Дополнительный префикс для фильтрации объектов внутри пути s3_url",
    ),
) -> dict[str, Any]:
    """Создаёт асинхронную задачу обработки видеофайлов и немедленно возвращает её номер.

    Файлы с адреса ``s3_url + prefix`` скачиваются в оперативную память (без сохранения
    на диск), транскрибируются Whisper, диаризуются pyannote, оцениваются по метрикам
    голоса и описываются Qwen. Итоговый JSON для каждого файла загружается обратно
    в исходный бакет под ключом ``output_data/<name>.json``.

    Args:
        s3_url: Адрес S3-хранилища или S3-compatible endpoint с бакетом.
        prefix: Дополнительный префикс внутри бакета для перебора объектов.

    Returns:
        JSON с полями ``task_id`` и ``status``.
    """
    db: Session = SessionLocal()
    try:
        task: Task = create_task(db, s3_url=s3_url, prefix=prefix)
        task_id: int = task.id
    finally:
        db.close()

    _submit_background_task(task_id)

    logger.info("Submitted task %d: s3_url=%r prefix=%r", task_id, s3_url, prefix)
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@app.get(
    "/tasks/{task_id}",
    summary="Получить статус и результат задачи",
    response_description="Подробная информация о задаче, включая логи и результаты",
)
async def get_task_status(task_id: int) -> dict[str, Any]:
    """Возвращает текущий статус задачи, прогресс, логи и результаты обработки.

    Args:
        task_id: Номер задачи, полученный при вызове ``/process``.

    Returns:
        JSON со всеми полями задачи.

    Raises:
        HTTPException 404: Если задача с таким ID не найдена.
    """
    db: Session = SessionLocal()
    try:
        task: Task | None = get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return _task_to_dict(task)
    finally:
        db.close()


@app.get(
    "/tasks",
    summary="Список всех задач",
    response_description="Краткая информация о последних задачах",
)
async def get_all_tasks(
    limit: int = Query(default=50, ge=1, le=500, description="Максимальное число записей"),
) -> list[dict[str, Any]]:
    """Возвращает список последних задач в порядке убывания ID (без логов и результатов).

    Args:
        limit: Количество записей (1–500, по умолчанию 50).

    Returns:
        Список словарей с кратким описанием каждой задачи.
    """
    db: Session = SessionLocal()
    try:
        tasks: list[Task] = list_tasks(db, limit=limit)
        return [_task_to_summary(t) for t in tasks]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime | None) -> str | None:
    """Форматирует datetime в ISO 8601 строку или возвращает None.

    Args:
        dt: Значение даты и времени или `None`.

    Returns:
        ISO 8601 строка или `None`.
    """
    return dt.isoformat() if dt is not None else None


def _task_to_dict(task: Task) -> dict[str, Any]:
    """Преобразует задачу в полное представление для API-ответа.

    Args:
        task: ORM-объект задачи.

    Returns:
        Словарь со всеми сериализуемыми полями задачи.
    """
    return {
        "id": task.id,
        "status": task.status,
        "s3_url": task.s3_url,
        "prefix": task.prefix,
        "progress": task.progress,
        "total": task.total,
        "current_file": task.current_file,
        "result": task.result,
        "error": task.error,
        "logs": task.logs,
        "created_at": _fmt_dt(task.created_at),
        "updated_at": _fmt_dt(task.updated_at),
    }


def _task_to_summary(task: Task) -> dict[str, Any]:
    """Преобразует задачу в краткое представление для списочного эндпоинта.

    Args:
        task: ORM-объект задачи.

    Returns:
        Словарь с краткой информацией о задаче без логов и детального результата.
    """
    return {
        "id": task.id,
        "status": task.status,
        "s3_url": task.s3_url,
        "prefix": task.prefix,
        "progress": task.progress,
        "total": task.total,
        "created_at": _fmt_dt(task.created_at),
        "updated_at": _fmt_dt(task.updated_at),
    }


_mount_sphinx_html(app)
