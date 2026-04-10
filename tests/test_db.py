"""Тесты для db.py."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

import db as db_module


@pytest.fixture
def isolated_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ModuleType:
    """Перезагружает модуль `db` с отдельной SQLite-базой.

    Args:
        monkeypatch: Фикстура monkeypatch.
        tmp_path: Временная директория pytest.

    Returns:
        Перезагруженный модуль `db`.
    """
    db_path: Path = tmp_path / "tasks.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reloaded = importlib.reload(db_module)
    return reloaded


class TestDbCrud:
    """Проверки CRUD-хелперов и ORM-модели задач."""

    def test_create_and_get_task(self, isolated_db: ModuleType) -> None:
        """Проверяет создание и чтение задачи из SQLite.

        Args:
            isolated_db: Перезагруженный модуль `db`.

        Returns:
            `None`.
        """
        session = isolated_db.SessionLocal()
        try:
            task = isolated_db.create_task(session, "s3://bucket", "input")
            loaded = isolated_db.get_task(session, task.id)
            assert loaded is not None
            assert loaded.s3_url == "s3://bucket"
            assert loaded.prefix == "input"
            assert loaded.status == isolated_db.TaskStatus.PENDING
        finally:
            session.close()

    def test_list_tasks_orders_desc(self, isolated_db: ModuleType) -> None:
        """Проверяет сортировку списка задач по убыванию ID.

        Args:
            isolated_db: Перезагруженный модуль `db`.

        Returns:
            `None`.
        """
        session = isolated_db.SessionLocal()
        try:
            first = isolated_db.create_task(session, "s3://bucket/1", "")
            second = isolated_db.create_task(session, "s3://bucket/2", "")
            tasks = isolated_db.list_tasks(session)
            assert tasks[0].id == second.id
            assert tasks[1].id == first.id
        finally:
            session.close()

    def test_update_task(self, isolated_db: ModuleType) -> None:
        """Проверяет обновление произвольных полей задачи.

        Args:
            isolated_db: Перезагруженный модуль `db`.

        Returns:
            `None`.
        """
        session = isolated_db.SessionLocal()
        try:
            task = isolated_db.create_task(session, "s3://bucket", "")
            isolated_db.update_task(
                session,
                task,
                status=isolated_db.TaskStatus.RUNNING,
                progress=3,
                total=10,
            )
            loaded = isolated_db.get_task(session, task.id)
            assert loaded is not None
            assert loaded.status == isolated_db.TaskStatus.RUNNING
            assert loaded.progress == 3
            assert loaded.total == 10
        finally:
            session.close()

    def test_append_log(self, isolated_db: ModuleType) -> None:
        """Проверяет добавление timestamped log в задачу.

        Args:
            isolated_db: Перезагруженный модуль `db`.

        Returns:
            `None`.
        """
        session = isolated_db.SessionLocal()
        try:
            task = isolated_db.create_task(session, "s3://bucket", "")
            isolated_db.append_log(session, task, "hello")
            loaded = isolated_db.get_task(session, task.id)
            assert loaded is not None
            assert loaded.logs is not None
            assert "hello" in loaded.logs[-1]
        finally:
            session.close()

    def test_get_task_returns_none_for_missing(
        self,
        isolated_db: ModuleType,
    ) -> None:
        """Проверяет возврат `None` для отсутствующей задачи.

        Args:
            isolated_db: Перезагруженный модуль `db`.

        Returns:
            `None`.
        """
        session = isolated_db.SessionLocal()
        try:
            assert isolated_db.get_task(session, 99999) is None
        finally:
            session.close()
