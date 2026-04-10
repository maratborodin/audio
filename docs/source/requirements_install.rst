.. _requirements-install:

Требования и установка
========================

Требования
----------

* macOS / Linux
* Python 3.11+
* установленный ``ffmpeg``
* локальная GGUF-модель Qwen и ``llama-cli`` для описания голосов
* доступ к HuggingFace для загрузки ``pyannote``

Установка
---------

Все команды ниже нужно выполнять из корня проекта.

1. Создать и активировать виртуальное окружение:

   .. code-block:: bash

      python3 -m venv venv
      source venv/bin/activate

2. Установить зависимости:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install httpx

   ``httpx`` нужен для тестов FastAPI через ``TestClient``.

3. Установить ``ffmpeg``.

   На macOS через Homebrew:

   .. code-block:: bash

      brew install ffmpeg

   Если используются динамические библиотеки Homebrew, при необходимости задайте:

   .. code-block:: bash

      export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"
