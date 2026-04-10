"""SQLAlchemy models and CRUD helpers for async task management."""
from __future__ import annotations

import enum
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./tasks.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite + потоки / отдельные процессы
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn: object, connection_record: object) -> None:
    """Включает WAL для SQLite, чтобы API и воркеры могли писать согласованно.

    Args:
        dbapi_conn: Низкоуровневое соединение SQLite.
        connection_record: Заглушка SQLAlchemy (не используется).
    """
    _ = connection_record
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Базовый декларативный класс SQLAlchemy для ORM-моделей проекта."""

    pass


class TaskStatus(str, enum.Enum):
    """Статус выполнения задачи обработки."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Task(Base):
    """Модель задачи обработки видеофайлов из S3.

    Attributes:
        id: Первичный ключ, автоинкрементный номер задачи.
        status: Текущий статус (pending / running / done / error).
        s3_url: Базовый URL S3-хранилища (например, s3://bucket/path).
        prefix: Дополнительный префикс для фильтрации объектов в бакете.
        progress: Количество обработанных файлов.
        total: Общее количество найденных файлов.
        current_file: S3-ключ файла, обрабатываемого в данный момент.
        result: JSON-массив с результатами по каждому файлу.
        error: Сообщение об ошибке при статусе ERROR.
        logs: JSON-массив строк с хронологией событий.
        created_at: Время создания задачи (UTC).
        updated_at: Время последнего обновления (UTC).
    """

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(16), default=TaskStatus.PENDING)
    s3_url: Mapped[str] = mapped_column(String(2048))
    prefix: Mapped[str] = mapped_column(String(1024), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    current_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    result: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    logs: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def create_task(db: Session, s3_url: str, prefix: str) -> Task:
    """Создаёт новую задачу в базе данных со статусом PENDING.

    Args:
        db: Сессия SQLAlchemy.
        s3_url: Базовый URL S3-хранилища.
        prefix: Префикс для фильтрации объектов.

    Returns:
        Созданный и закоммиченный объект Task.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: При ошибках вставки или commit в базе данных.
    """
    task: Task = Task(
        status=TaskStatus.PENDING,
        s3_url=s3_url,
        prefix=prefix,
        progress=0,
        total=0,
        logs=[],
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: int) -> Task | None:
    """Возвращает задачу по её ID или None.

    Args:
        db: Сессия SQLAlchemy.
        task_id: Идентификатор задачи.

    Returns:
        Объект Task или None, если задача не найдена.
    """
    return db.query(Task).filter(Task.id == task_id).first()


def list_tasks(db: Session, limit: int = 100) -> list[Task]:
    """Возвращает список задач в порядке убывания ID.

    Args:
        db: Сессия SQLAlchemy.
        limit: Максимальное число возвращаемых записей.

    Returns:
        Список объектов Task.
    """
    return db.query(Task).order_by(Task.id.desc()).limit(limit).all()


def update_task(db: Session, task: Task, **fields: Any) -> Task:
    """Обновляет произвольные поля задачи и обновляет updated_at.

    Args:
        db: Сессия SQLAlchemy.
        task: Объект Task для обновления.
        **fields: Поля и их новые значения.

    Returns:
        Обновлённый объект Task.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: При ошибках commit в базе данных.
    """
    for key, value in fields.items():
        setattr(task, key, value)
    task.updated_at = datetime.now(timezone.utc)
    db.commit()
    return task


def append_log(db: Session, task: Task, message: str) -> None:
    """Добавляет строку лога с временной меткой в task.logs и сохраняет.

    Args:
        db: Сессия SQLAlchemy.
        task: Объект Task.
        message: Сообщение лога.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: При ошибках commit в базе данных.
    """
    logs: list[str] = list(task.logs or [])
    timestamp: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logs.append(f"[{timestamp}] {message}")
    task.logs = logs
    task.updated_at = datetime.now(timezone.utc)
    db.commit()
