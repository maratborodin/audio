# Audio Processing Service

Сервис для асинхронной обработки видеофайлов:

- извлечение аудио из видео через `ffmpeg`
- транскрибация речи через `Whisper`
- диаризация спикеров через `pyannote`
- оценка голосовых метрик
- генерация текстового описания голоса через локальную Qwen GGUF-модель
- загрузка итогового JSON обратно в S3-совместимое хранилище

Проект поддерживает:

- локальную обработку директории с видеофайлами через модуль `process.py`
- web API на FastAPI (`main.py`)
- развёртывание API в Docker и запуск тяжёлой обработки каждой задачи в **отдельном одноразовом контейнере** (см. `docker-compose.yml` и переменные `USE_DOCKER_WORKERS`, `WORKER_IMAGE`)

## Возможности

- приём задач на обработку по S3 URL и префиксу
- асинхронное выполнение задач в фоне (поток процесса API или одноразовый Docker-контейнер на задачу)
- хранение статуса, прогресса, логов и результатов в `SQLite` через `SQLAlchemy`
- работа с AWS S3 и S3-compatible storage, включая MinIO
- сохранение результата в тот же бакет под префиксом `output_data/`
- покрытие тестами выше 80%

## Структура проекта

- `main.py` — FastAPI сервер, API-эндпоинты, постановка задач (поток или Docker)
- `task_runner.py` — выполнение задачи по `task_id` (S3, Whisper, выгрузка JSON); CLI: `python -m task_runner --task-id N`
- `docker_runner.py` — запуск одноразового контейнера-воркера через Docker CLI
- `process.py` — основная логика обработки аудио и видео
- `diarization.py` — диаризация спикеров
- `voice_assessment.py` — расчёт акустических метрик голоса
- `qwen.py` — генерация текстового описания голоса через `llama-cli`
- `s3_client.py` — работа с S3 / MinIO через `boto3`
- `db.py` — модели и CRUD-хелперы для `SQLite`
- `tests/` — набор unit-тестов
- `Dockerfile`, `docker-compose.yml` — образ приложения и сервис API в Compose

## Как работает обработка

### Web-режим

1. Клиент вызывает `GET /process` с параметрами `s3_url` и `prefix`.
2. Сервер создаёт запись о задаче в SQLite.
3. Запуск обработки (один из вариантов):

   - **Локально** (`USE_DOCKER_WORKERS` не задан или `0`): в фоновом потоке вызывается `task_runner.run_processing_task(task_id)` в том же процессе, что и API.
   - **Docker** (`USE_DOCKER_WORKERS=1` и задан `WORKER_IMAGE`): API выполняет `docker run` (отсоединённый контейнер) с командой `python -m task_runner --task-id <id>`. Каждая задача получает **свой** контейнер; после завершения контейнер удаляется (`--rm`). Том с SQLite и данными монтируется в воркер так же, как в сервис API (см. `docker-compose.yml`).

4. Внутри воркера по очереди: список объектов S3 → скачивание видео в память → `ffmpeg` → Whisper → диаризация → метрики → описание голоса.

5. Результат сохраняется в тот же бакет по ключу:

```text
output_data/<имя_видеофайла>.json
```

6. Статус и логи доступны через API.

### Локальный режим

Локальный batch-режим вызывается через `process.py` и обрабатывает все видеофайлы из директории, сохраняя результаты локально.

## Требования

- macOS / Linux
- Python `3.11+`
- установленный `ffmpeg`
- локальная GGUF-модель Qwen и `llama-cli` для описания голосов
- доступ к HuggingFace для `pyannote`

## Установка

Все команды ниже нужно выполнять из корня проекта.

### 1. Создать и активировать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
pip install httpx
```

`httpx` нужен для тестов FastAPI через `TestClient`.

### 3. Установить `ffmpeg`

На macOS через Homebrew:

```bash
brew install ffmpeg
```

Если используются динамические библиотеки Homebrew, при необходимости задайте:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"
```

## Настройка переменных окружения

Создайте файл `.env` в корне проекта.

Пример (вместо многоточий укажите свои значения):

```env
TOKEN=<…>
S3_ENDPOINT_URL=<…>
AWS_ACCESS_KEY_ID=<…>
AWS_SECRET_ACCESS_KEY=<…>
DATA_PATH=./output_data
DATABASE_URL=sqlite:///./tasks.db
```

Описание переменных:

- `TOKEN` — доступ к моделям HuggingFace (в т.ч. `pyannote/speaker-diarization-community-1`)
- `S3_ENDPOINT_URL` — адрес S3-compatible API (например, локальный MinIO)
- `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` — пара параметров для аутентификации в хранилище (задаются провайдером или администратором)
- `DATA_PATH` — локальная директория для промежуточных/локальных JSON
- `DATABASE_URL` — путь к SQLite базе задач
- `USE_DOCKER_WORKERS` — `1` для режима одноразовых Docker-контейнеров (в Compose задаётся автоматически)
- `WORKER_IMAGE` — имя образа для `docker run` (в Compose совпадает с образом API)
- `TASK_VOLUME_NAME` — имя Docker volume, куда смонтирован путь с БД (например `audio_task_data` для проекта Compose `audio`)
- `DOCKER_NETWORK` — сеть Compose (например `audio_default`), чтобы воркер видел MinIO/S3 по имени сервиса

Если используете MinIO Client (`mc`), при совпадении endpoint учётные данные могут подхватываться из локального конфига клиента.

## Запуск сервера

Перед запуском нужно загрузить переменные из `.env`:

```bash
set -a
source .env
set +a
```

Запуск FastAPI:

```bash
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Для разработки с autoreload:

```bash
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

В браузере:

- **OpenAPI (Swagger UI):** `http://localhost:8000/api/docs`
- **ReDoc:** `http://localhost:8000/api/redoc`
- **Документация проекта (Sphinx):** `http://localhost:8000/docs/` — сначала выполните `./venv/bin/sphinx-build -b html docs/source docs/build/html`. Если сборки нет, префикс `/docs/` не раздаётся (см. предупреждение в логе при старте).

## API

### `GET /process`

Создаёт асинхронную задачу обработки.

Параметры:

- `s3_url` — адрес бакета или endpoint-style URL
- `prefix` — префикс внутри бакета

Пример:

```bash
curl "http://localhost:8000/process?s3_url=s3://127.0.0.1:9000/test-bucket/&prefix=input_video/"
```

Пример ответа:

```json
{
  "task_id": 1,
  "status": "pending"
}
```

### `GET /tasks/{task_id}`

Возвращает:

- статус задачи
- текущий прогресс
- логи
- результат по каждому обработанному файлу

Пример:

```bash
curl "http://localhost:8000/tasks/1"
```

### `GET /tasks`

Возвращает список последних задач:

```bash
curl "http://localhost:8000/tasks"
```

## Формат результата

Итоговый JSON содержит:

- `transcription`
- `diarization`
- для каждого спикера:
  - `segments`
  - `voice_metrics`
  - `voice_description`

Результат сохраняется в исходный бакет под ключом:

```text
output_data/<имя_исходного_видео>.json
```

Например:

```text
input_video/interview.webm -> output_data/interview.json
```

## Локальный запуск batch-обработки

Если нужно обработать директорию локально без web API:

```bash
source venv/bin/activate
python process.py
```

По умолчанию ожидается директория:

```text
./input_video/
```

Результаты локального режима сохраняются в:

- `./output_audio`
- `./output_data`

## Запуск тестов

Все тесты нужно запускать из `venv`.

Обычный запуск:

```bash
source venv/bin/activate
pytest
```

Запуск без активации:

```bash
./venv/bin/python -m pytest
```

Параллельный запуск:

```bash
pytest -n 8
```

Полный запуск с покрытием:

```bash
pytest --override-ini addopts='-v --tb=short -q -n 0' --cov=. --cov-report=term-missing --cov-fail-under=80
```

Текущий проект покрыт тестами выше 80%.

Подробнее: [tests/README.md](tests/README.md)

## База данных задач

По умолчанию используется:

```text
sqlite:///./tasks.db
```

В таблице `tasks` хранятся:

- `id`
- `status`
- `s3_url`
- `prefix`
- `progress`
- `total`
- `current_file`
- `result`
- `error`
- `logs`
- `created_at`
- `updated_at`

## Особенности работы с S3 / MinIO

Поддерживаются форматы:

- `s3://bucket/prefix`
- `s3://127.0.0.1:9000/bucket/prefix`
- `http://127.0.0.1:9000/bucket/prefix`
- `https://bucket.s3.amazonaws.com/prefix`

Для MinIO обычно используется API порт `9000`, а не web-console порт `9001`.

Пример проверки содержимого бакета через `mc`:

```bash
./mc ls local/test-bucket/input_video
```

Пример проверки результатов:

```bash
./mc ls local/test-bucket/output_data
```

## Частые проблемы

### `NoCredentialsError`

Проверьте:

- загружен ли `.env`
- заданы ли в окружении параметры аутентификации для S3
- совпадает ли `S3_ENDPOINT_URL` с реальным endpoint storage

### `0 objects found`

Проверьте:

- правильный ли `prefix`
- что в именах объектов в бакете нет ведущего `/`
- что используется правильный endpoint и порт

### Ошибки `ffmpeg` / `torchcodec`

Если есть проблемы с динамическими библиотеками `ffmpeg`, проверьте установку `ffmpeg` и переменную:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"
```

### Не работает Qwen

Проверьте:

- доступность `llama-cli`
- наличие GGUF-модели
- корректность пути `model_path` в `qwen.py`

## Развёртывание

### Локально (без Docker)

1. Клонировать проект.
2. Создать `venv`, установить зависимости и `ffmpeg`.
3. Подготовить `.env`.
4. Убедиться, что доступны S3 / MinIO, HuggingFace, `llama-cli`, GGUF-модель.
5. Запустить сервер (обработка задач — в фоновом потоке, без отдельных контейнеров):

```bash
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

1. Установить [Docker](https://docs.docker.com/get-docker/) и Docker Compose v2.
2. В корне проекта задать переменные (через `.env` рядом с `docker-compose.yml` или экспорт): как минимум `TOKEN` и при необходимости параметры S3.
3. Собрать и поднять сервис API:

```bash
docker compose build
docker compose up -d api
```

Сервис `api` монтирует сокет Docker (`/var/run/docker.sock`) и общий том `task_data` в `/data`. При каждом `GET /process` создаётся **одноразовый** контейнер из того же образа (`WORKER_IMAGE`), который выполняет `python -m task_runner --task-id …` и завершается. База SQLite и `DATA_PATH` лежат на общем томе, для БД включён режим WAL.

Требования: образ воркера должен быть доступен движку Docker (локально собран как `audio-api:latest`). Доступ к S3/MinIO из контейнеров настройте через `S3_ENDPOINT_URL` (для сервиса в той же сети Compose — имя хоста вида `http://minio:9000`).

## Лицензирование и безопасность

- не коммитьте секреты и учётные данные в репозиторий
- `.env` должен храниться только локально или в секрет-хранилище
- для production рекомендуется отдельная БД, отдельные учётные данные для хранилища и запуск под process manager
