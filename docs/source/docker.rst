.. _docker-deployment:

Docker и одноразовые воркеры
============================

Образ приложения собирается из ``Dockerfile`` (Python 3.11, ``ffmpeg``, зависимости из ``requirements.txt``).

Сервис ``api`` в ``docker-compose.yml``:

* публикует порт ``8000``;
* монтирует именованный том ``task_data`` в ``/data`` (SQLite и ``DATA_PATH``);
* монтирует сокет Docker для запуска воркеров: ``/var/run/docker.sock``;
* задаёт ``USE_DOCKER_WORKERS=1``, ``WORKER_IMAGE``, ``TASK_VOLUME_NAME``, ``DOCKER_NETWORK``.

При создании задачи (``GET /process``) API вызывает ``docker run --rm -d`` с образом воркера и командой ``python -m task_runner --task-id <id>``. Каждая задача выполняется в **отдельном** контейнере; после выхода процесса контейнер удаляется.

Локальный запуск без Docker по-прежнему использует фоновый поток и ``USE_DOCKER_WORKERS=0`` (или не заданную переменную).

Подробности переменных окружения см. в :ref:`environment`.
