.. _deployment:

Развёртывание
=============

Минимальный сценарий деплоя:

#. Клонировать проект.
#. Создать ``venv``.
#. Установить зависимости.
#. Установить ``ffmpeg``.
#. Подготовить ``.env``.
#. Убедиться, что доступны:

   * S3 / MinIO
   * доступ к HuggingFace для загрузки моделей
   * ``llama-cli``
   * GGUF-модель

#. Запустить сервер:

   .. code-block:: bash

      ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

Docker Compose
--------------

См. отдельно :ref:`docker-deployment`. Кратко: ``docker compose build`` и ``docker compose up -d api``; обработка каждой задачи — в отдельном контейнере-воркере.
