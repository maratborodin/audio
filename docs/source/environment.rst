.. _environment:

Настройка переменных окружения
==============================

Создайте файл ``.env`` в корне проекта.

Пример (вместо многоточий укажите свои значения):

.. code-block:: env

   TOKEN=<…>
   S3_ENDPOINT_URL=<…>
   AWS_ACCESS_KEY_ID=<…>
   AWS_SECRET_ACCESS_KEY=<…>
   DATA_PATH=./output_data
   DATABASE_URL=sqlite:///./tasks.db

Описание переменных:

* ``TOKEN`` — доступ к моделям HuggingFace (в т.ч. ``pyannote/speaker-diarization-community-1``)
* ``S3_ENDPOINT_URL`` — адрес S3-compatible API (например, локальный MinIO)
* ``AWS_ACCESS_KEY_ID`` и ``AWS_SECRET_ACCESS_KEY`` — пара параметров для аутентификации в хранилище (задаются провайдером или администратором)
* ``DATA_PATH`` — локальная директория для промежуточных/локальных JSON
* ``DATABASE_URL`` — путь к SQLite базе задач
* ``USE_DOCKER_WORKERS`` — ``1`` для запуска воркеров через ``docker run`` (в Compose обычно включено)
* ``WORKER_IMAGE`` — имя образа Docker для воркера (совпадает с образом API)
* ``TASK_VOLUME_NAME`` — имя тома Docker с данными ``/data`` (например ``audio_task_data``)
* ``DOCKER_NETWORK`` — сеть Docker Compose (например ``audio_default``)

Если используете MinIO Client (``mc``), при совпадении endpoint учётные данные могут подхватываться из локального конфига клиента.
