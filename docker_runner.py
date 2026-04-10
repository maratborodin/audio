"""Запуск одноразового Docker-контейнера для обработки задачи ``task_id``."""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from typing import Sequence

logger: logging.Logger = logging.getLogger(__name__)

# Переменные окружения, пробрасываемые в контейнер воркера (без лишних ключей хоста).
_WORKER_ENV_KEYS: tuple[str, ...] = (
    "TOKEN",
    "DATABASE_URL",
    "DATA_PATH",
    "S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)


def _bool_env(name: str, default: bool = False) -> bool:
    """Возвращает булево значение переменной окружения.

    Args:
        name: Имя переменной.
        default: Значение, если переменная не задана.

    Returns:
        ``True`` для ``1``, ``true``, ``yes`` (без учёта регистра).
    """
    raw: str | None = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def use_docker_workers() -> bool:
    """Проверяет, нужно ли ставить задачу в одноразовый контейнер.

    Returns:
        ``True``, если ``USE_DOCKER_WORKERS`` включён и задан ``WORKER_IMAGE``.
    """
    if not _bool_env("USE_DOCKER_WORKERS", default=False):
        return False
    image: str | None = os.environ.get("WORKER_IMAGE", "").strip()
    return bool(image)


def build_docker_run_command(task_id: int) -> list[str]:
    """Собирает команду ``docker run`` для воркера с заданным ``task_id``.

    Args:
        task_id: Номер задачи в БД.

    Returns:
        Список аргументов для ``subprocess``.

    Raises:
        RuntimeError: Если не задан ``WORKER_IMAGE``.
    """
    image: str = os.environ.get("WORKER_IMAGE", "").strip()
    if not image:
        raise RuntimeError("WORKER_IMAGE must be set when USE_DOCKER_WORKERS=1")

    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "-d",
    ]

    network: str | None = os.environ.get("DOCKER_NETWORK", "").strip()
    if network:
        cmd.extend(["--network", network])

    volume_name: str | None = os.environ.get("TASK_VOLUME_NAME", "").strip()
    if volume_name:
        mount_target: str = "/data"
        cmd.extend(["-v", f"{volume_name}:{mount_target}"])

    key: str
    for key in _WORKER_ENV_KEYS:
        val: str | None = os.environ.get(key)
        if val is not None and val != "":
            cmd.extend(["-e", f"{key}={val}"])

    cmd.append(image)
    cmd.extend(["python", "-m", "task_runner", "--task-id", str(task_id)])
    return cmd


def spawn_worker_container(task_id: int) -> None:
    """Стартует отсоединённый контейнер воркера; не ждёт завершения.

    Args:
        task_id: Идентификатор задачи.

    Raises:
        RuntimeError: При ошибке запуска ``docker run``.
    """
    try:
        cmd: list[str] = build_docker_run_command(task_id)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise

    logger.info("Starting worker container for task %d: %s", task_id, shlex.join(cmd))
    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        msg: str = "docker CLI not found; install Docker or set USE_DOCKER_WORKERS=0"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        err: str = (exc.stderr or exc.stdout or "").strip()
        logger.error("docker run failed for task %d: %s", task_id, err)
        raise RuntimeError(f"docker run failed: {err}") from exc

    cid: str = (proc.stdout or "").strip()
    if cid:
        logger.info("Worker container started: %s", cid)
