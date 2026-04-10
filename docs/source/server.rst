.. _server:

Запуск сервера
==============

Перед запуском нужно загрузить переменные из ``.env``:

.. code-block:: bash

   set -a
   source .env
   set +a

Запуск FastAPI:

.. code-block:: bash

   ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

Для разработки с autoreload:

.. code-block:: bash

   ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Документация в браузере
------------------------

* **OpenAPI (Swagger UI):** ``http://localhost:8000/api/docs``
* **ReDoc:** ``http://localhost:8000/api/redoc``
* **Справочник Sphinx (HTML):** ``http://localhost:8000/docs/`` — предварительно соберите проект:

  .. code-block:: bash

     ./venv/bin/sphinx-build -b html docs/source docs/build/html

  Если каталог ``docs/build/html`` отсутствует, раздача по ``/docs/`` не подключается (см. логи сервера).
