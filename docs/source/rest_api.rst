.. _rest-api:

HTTP API
========

``GET /process``
----------------

Создаёт асинхронную задачу обработки.

Параметры:

* ``s3_url`` — адрес бакета или endpoint-style URL
* ``prefix`` — префикс внутри бакета

Пример:

.. code-block:: bash

   curl "http://localhost:8000/process?s3_url=s3://127.0.0.1:9000/test-bucket/&prefix=input_video/"

Пример ответа:

.. code-block:: json

   {
     "task_id": 1,
     "status": "pending"
   }

``GET /tasks/{task_id}``
------------------------

Возвращает:

* статус задачи
* текущий прогресс
* логи
* результат по каждому обработанному файлу

Пример:

.. code-block:: bash

   curl "http://localhost:8000/tasks/1"

``GET /tasks``
--------------

Возвращает список последних задач:

.. code-block:: bash

   curl "http://localhost:8000/tasks"
