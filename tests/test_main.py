"""Тесты для FastAPI-сервера `main.py`."""
from __future__ import annotations

import importlib
from datetime import datetime
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Импортирует `main` с лёгкой заглушкой модуля `process`.

    Args:
        monkeypatch: Фикстура monkeypatch.

    Returns:
        Перезагруженный модуль `main`.
    """
    fake_process: ModuleType = ModuleType("process")
    fake_process.load_whisper_pipeline = lambda: object()
    fake_process.process_video_bytes = lambda **kwargs: {
        "transcription": "stub",
        "diarization": {},
    }
    monkeypatch.setitem(sys.modules, "process", fake_process)

    import main as imported_main

    return importlib.reload(imported_main)


@pytest.fixture
def task_runner_module(main_module: ModuleType) -> ModuleType:
    """Возвращает модуль ``task_runner`` после загрузки ``main`` (с подменой ``process``).

    Args:
        main_module: Загруженный ``main`` (используется для порядка инициализации).

    Returns:
        Модуль ``task_runner``.
    """
    _ = main_module
    return importlib.import_module("task_runner")


@pytest.fixture
def client(main_module: ModuleType) -> TestClient:
    """Возвращает тестовый HTTP-клиент для FastAPI-приложения.

    Returns:
        Экземпляр `TestClient`.
    """
    return TestClient(main_module.app)


class TestMainHelpers:
    """Проверки вспомогательных функций web-сервера."""

    def test_join_s3_prefixes(self, main_module: ModuleType) -> None:
        """Проверяет нормализацию префиксов S3.

        Returns:
            `None`.
        """
        assert main_module._join_s3_prefixes("videos/", "/input/") == "videos/input"
        assert main_module._join_s3_prefixes("", "input/") == "input"
        assert main_module._join_s3_prefixes("videos", "") == "videos"

    def test_build_output_key(self, main_module: ModuleType) -> None:
        """Проверяет формирование ключа результата в S3.

        Returns:
            `None`.
        """
        assert (
            main_module._build_output_key("input_video/sample.webm")
            == "output_data/sample.json"
        )

    def test_fmt_dt(self, main_module: ModuleType) -> None:
        """Проверяет сериализацию datetime в ISO-формат.

        Returns:
            `None`.
        """
        value: datetime = datetime(2024, 1, 2, 3, 4, 5)
        assert main_module._fmt_dt(value).startswith("2024-01-02T03:04:05")
        assert main_module._fmt_dt(None) is None

    def test_task_serializers(self, main_module: ModuleType) -> None:
        """Проверяет сериализацию полной и краткой карточки задачи.

        Returns:
            `None`.
        """
        task: SimpleNamespace = SimpleNamespace(
            id=1,
            status="done",
            s3_url="s3://bucket",
            prefix="input",
            progress=1,
            total=1,
            current_file=None,
            result=[{"status": "ok"}],
            error=None,
            logs=["ok"],
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            updated_at=datetime(2024, 1, 1, 1, 0, 0),
        )
        full: dict[str, object] = main_module._task_to_dict(task)
        short: dict[str, object] = main_module._task_to_summary(task)
        assert full["id"] == 1
        assert full["result"] == [{"status": "ok"}]
        assert short["id"] == 1
        assert "result" not in short


class TestMainEndpoints:
    """Проверки HTTP-эндпоинтов FastAPI-сервера."""

    def test_process_endpoint_creates_task(
        self,
        main_module: ModuleType,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет создание задачи и вызов постановки в фон (_submit_background_task).

        Args:
            client: Тестовый HTTP-клиент.
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        submitted: list[tuple[object, ...]] = []

        def fake_create_task(
            db: object,
            s3_url: str,
            prefix: str,
        ) -> SimpleNamespace:
            _ = db
            return SimpleNamespace(id=42, s3_url=s3_url, prefix=prefix)

        def fake_submit(task_id: int) -> None:
            submitted.append((main_module._run_processing_task, task_id))

        monkeypatch.setattr(main_module, "SessionLocal", lambda: MagicMock())
        monkeypatch.setattr(main_module, "create_task", fake_create_task)
        monkeypatch.setattr(main_module, "_submit_background_task", fake_submit)

        response = client.get(
            "/process",
            params={"s3_url": "s3://bucket", "prefix": "input_video/"},
        )

        assert response.status_code == 200
        assert response.json()["task_id"] == 42
        assert submitted[0][0] is main_module._run_processing_task
        assert submitted[0][1] == 42

    def test_get_task_status_returns_404(
        self,
        main_module: ModuleType,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет 404 для несуществующей задачи.

        Args:
            client: Тестовый HTTP-клиент.
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        monkeypatch.setattr(main_module, "SessionLocal", lambda: MagicMock())
        monkeypatch.setattr(main_module, "get_task", lambda db, task_id: None)

        response = client.get("/tasks/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_task_status_returns_payload(
        self,
        main_module: ModuleType,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет успешную выдачу состояния задачи.

        Args:
            client: Тестовый HTTP-клиент.
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        task: SimpleNamespace = SimpleNamespace(
            id=5,
            status="running",
            s3_url="s3://bucket",
            prefix="input",
            progress=1,
            total=2,
            current_file="input/a.webm",
            result=None,
            error=None,
            logs=["step"],
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            updated_at=datetime(2024, 1, 1, 0, 1, 0),
        )
        monkeypatch.setattr(main_module, "SessionLocal", lambda: MagicMock())
        monkeypatch.setattr(main_module, "get_task", lambda db, task_id: task)

        response = client.get("/tasks/5")

        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert response.json()["current_file"] == "input/a.webm"

    def test_get_all_tasks_returns_summary(
        self,
        main_module: ModuleType,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет выдачу краткого списка задач.

        Args:
            client: Тестовый HTTP-клиент.
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        tasks: list[SimpleNamespace] = [
            SimpleNamespace(
                id=1,
                status="done",
                s3_url="s3://bucket",
                prefix="input",
                progress=1,
                total=1,
                created_at=datetime(2024, 1, 1, 0, 0, 0),
                updated_at=datetime(2024, 1, 1, 0, 2, 0),
            )
        ]
        monkeypatch.setattr(main_module, "SessionLocal", lambda: MagicMock())
        monkeypatch.setattr(main_module, "list_tasks", lambda db, limit=50: tasks)

        response = client.get("/tasks")

        assert response.status_code == 200
        assert response.json()[0]["id"] == 1
        assert "result" not in response.json()[0]


class TestRunProcessingTask:
    """Проверки исполнителя ``task_runner.run_processing_task`` (алиас ``_run_processing_task``)."""

    def test_no_task_found(
        self,
        main_module: ModuleType,
        task_runner_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет ранний выход, если задача отсутствует в БД.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_session: MagicMock = MagicMock()
        monkeypatch.setattr(task_runner_module, "SessionLocal", lambda: fake_session)
        monkeypatch.setattr(task_runner_module, "get_task", lambda db, task_id: None)

        main_module._run_processing_task(123)

        fake_session.close.assert_called_once()

    def test_no_keys_marks_task_done(
        self,
        main_module: ModuleType,
        task_runner_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет ветку завершения задачи без найденных файлов.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_session: MagicMock = MagicMock()
        task: SimpleNamespace = SimpleNamespace(
            id=1,
            s3_url="s3://bucket",
            prefix="input",
        )
        updates: list[dict[str, object]] = []
        logs: list[str] = []

        def fake_update_task(
            db: object,
            task_obj: object,
            **fields: object,
        ) -> object:
            _ = db, task_obj
            updates.append(fields)
            return task

        monkeypatch.setattr(task_runner_module, "SessionLocal", lambda: fake_session)
        monkeypatch.setattr(task_runner_module, "get_task", lambda db, task_id: task)
        monkeypatch.setattr(task_runner_module, "update_task", fake_update_task)
        monkeypatch.setattr(task_runner_module, "append_log", lambda db, task_obj, msg: logs.append(msg))
        monkeypatch.setattr(task_runner_module, "parse_s3_url", lambda s3_url: ("bucket", "", None))
        monkeypatch.setattr(task_runner_module, "list_video_keys", lambda bucket, prefix, endpoint_url=None: [])

        main_module._run_processing_task(1)

        assert any(update.get("status") == "running" for update in updates)
        assert any(update.get("status") == "done" for update in updates)
        assert "No files to process. Task done." in logs[-1]

    def test_successful_task_uploads_output(
        self,
        main_module: ModuleType,
        task_runner_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет успешную обработку файла и загрузку JSON в S3.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_session: MagicMock = MagicMock()
        task: SimpleNamespace = SimpleNamespace(
            id=2,
            s3_url="s3://bucket",
            prefix="input",
        )
        uploads: list[tuple[str, str, bytes]] = []
        updates: list[dict[str, object]] = []

        monkeypatch.setattr(task_runner_module, "SessionLocal", lambda: fake_session)
        monkeypatch.setattr(task_runner_module, "get_task", lambda db, task_id: task)
        monkeypatch.setattr(
            task_runner_module,
            "update_task",
            lambda db, task_obj, **fields: updates.append(fields) or task,
        )
        monkeypatch.setattr(task_runner_module, "append_log", lambda db, task_obj, msg: None)
        monkeypatch.setattr(task_runner_module, "parse_s3_url", lambda s3_url: ("bucket", "", "http://s3"))
        monkeypatch.setattr(
            task_runner_module,
            "list_video_keys",
            lambda bucket, prefix, endpoint_url=None: ["input_video/a.webm"],
        )
        monkeypatch.setattr(
            task_runner_module,
            "download_to_memory",
            lambda bucket, key, endpoint_url=None: b"video-bytes",
        )
        monkeypatch.setattr(task_runner_module, "load_whisper_pipeline", lambda: object())
        monkeypatch.setattr(
            task_runner_module,
            "process_video_bytes",
            lambda **kwargs: {"transcription": "hello", "diarization": {}},
        )
        monkeypatch.setattr(
            task_runner_module,
            "upload_json_bytes",
            lambda bucket, key, payload, endpoint_url=None: uploads.append(
                (bucket, key, payload)
            ),
        )

        main_module._run_processing_task(2)

        assert uploads
        assert uploads[0][0] == "bucket"
        assert uploads[0][1] == "output_data/a.json"
        assert b'"transcription": "hello"' in uploads[0][2]
        assert any(update.get("progress") == 1 for update in updates)
        assert any(update.get("status") == "done" for update in updates)

    def test_processing_error_is_collected(
        self,
        main_module: ModuleType,
        task_runner_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Проверяет сохранение ошибки файла в `result`, не падая всей задачей.

        Args:
            monkeypatch: Фикстура monkeypatch.

        Returns:
            `None`.
        """
        fake_session: MagicMock = MagicMock()
        task: SimpleNamespace = SimpleNamespace(
            id=3,
            s3_url="s3://bucket",
            prefix="input",
        )
        updates: list[dict[str, object]] = []

        monkeypatch.setattr(task_runner_module, "SessionLocal", lambda: fake_session)
        monkeypatch.setattr(task_runner_module, "get_task", lambda db, task_id: task)
        monkeypatch.setattr(
            task_runner_module,
            "update_task",
            lambda db, task_obj, **fields: updates.append(fields) or task,
        )
        monkeypatch.setattr(task_runner_module, "append_log", lambda db, task_obj, msg: None)
        monkeypatch.setattr(task_runner_module, "parse_s3_url", lambda s3_url: ("bucket", "", None))
        monkeypatch.setattr(
            task_runner_module,
            "list_video_keys",
            lambda bucket, prefix, endpoint_url=None: ["input_video/a.webm"],
        )
        monkeypatch.setattr(
            task_runner_module,
            "download_to_memory",
            lambda bucket, key, endpoint_url=None: b"video",
        )
        monkeypatch.setattr(task_runner_module, "load_whisper_pipeline", lambda: object())

        def raise_processing_error(**kwargs: object) -> dict[str, object]:
            _ = kwargs
            raise RuntimeError("boom")

        monkeypatch.setattr(task_runner_module, "process_video_bytes", raise_processing_error)

        main_module._run_processing_task(3)

        done_updates: list[dict[str, object]] = [
            update for update in updates if update.get("status") == "done"
        ]
        assert done_updates
        result_payload = done_updates[-1]["result"]
        assert isinstance(result_payload, list)
        assert result_payload[0]["status"] == "error"
