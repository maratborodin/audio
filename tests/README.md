# Тесты

Все команды нужно выполнять в окружении из папки `venv` (из корня проекта):

```bash
source venv/bin/activate
```

Либо вызывать интерпретатор и pytest явно через venv:

```bash
./venv/bin/python -m pytest ...
```

Запуск всех тестов:

```bash
pytest
# или
./venv/bin/python -m pytest
```

Параллельный запуск на 8 потоков (pytest-xdist):

```bash
pytest -n 8
```

Запуск с отчётом покрытия (не менее 80%):

```bash
# из корня проекта
pytest --cov=. --cov-report=term-missing --cov-fail-under=80
```

Покрытие в параллельном режиме:

```bash
pytest -n 8 --cov=. --cov-report=term-missing --cov-fail-under=80
```

Запуск по модулям (без тяжёлых зависимостей main):

```bash
pytest tests/test_qwen.py tests/test_diarization.py tests/test_voice_assessment.py -v
pytest tests/test_main.py -v
```

Внешние вызовы замоканы: `subprocess` (llama-cli), `pyannote` Pipeline, `soundfile`, `ffmpeg`, `torch`/`transformers` в main.
