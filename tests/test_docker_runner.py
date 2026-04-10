"""Тесты для модуля `docker_runner`."""
from __future__ import annotations

import pytest

from docker_runner import build_docker_run_command, use_docker_workers


class TestDockerRunner:
    """Проверки сборки команды `docker run` и флага режима."""

    def test_use_docker_workers_false_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Без переменных режим Docker-воркеров выключен.

        Returns:
            `None`.
        """
        monkeypatch.delenv("USE_DOCKER_WORKERS", raising=False)
        monkeypatch.delenv("WORKER_IMAGE", raising=False)
        assert use_docker_workers() is False

    def test_use_docker_workers_true_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """При USE_DOCKER_WORKERS и WORKER_IMAGE режим включён.

        Returns:
            `None`.
        """
        monkeypatch.setenv("USE_DOCKER_WORKERS", "1")
        monkeypatch.setenv("WORKER_IMAGE", "audio-api:latest")
        assert use_docker_workers() is True

    def test_build_docker_run_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Команда содержит образ, том, сеть и `--task-id`.

        Returns:
            `None`.
        """
        monkeypatch.setenv("WORKER_IMAGE", "myimg:1")
        monkeypatch.setenv("TASK_VOLUME_NAME", "vol1")
        monkeypatch.setenv("DOCKER_NETWORK", "net1")
        monkeypatch.setenv("DATABASE_URL", "sqlite:////data/tasks.db")
        monkeypatch.setenv("DATA_PATH", "/data/output")
        cmd: list[str] = build_docker_run_command(7)
        assert "docker" in cmd[0]
        assert "myimg:1" in cmd
        assert "--network" in cmd
        assert "net1" in cmd
        assert "-v" in cmd
        assert "vol1:/data" in cmd
        assert "--task-id" in cmd
        assert cmd[-1] == "7"

    def test_build_docker_run_raises_without_image(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Без WORKER_IMAGE — исключение.

        Returns:
            `None`.
        """
        monkeypatch.setenv("USE_DOCKER_WORKERS", "1")
        monkeypatch.delenv("WORKER_IMAGE", raising=False)
        with pytest.raises(RuntimeError, match="WORKER_IMAGE"):
            build_docker_run_command(1)
