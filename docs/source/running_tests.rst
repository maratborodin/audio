.. _running-tests:

Запуск тестов
=============

Все тесты нужно запускать из ``venv``.

Обычный запуск:

.. code-block:: bash

   source venv/bin/activate
   pytest

Запуск без активации:

.. code-block:: bash

   ./venv/bin/python -m pytest

Параллельный запуск:

.. code-block:: bash

   pytest -n 8

Полный запуск с покрытием:

.. code-block:: bash

   pytest --override-ini addopts='-v --tb=short -q -n 0' --cov=. --cov-report=term-missing --cov-fail-under=80

Текущий проект покрыт тестами выше 80%.

Дополнительные инструкции см. в файле ``tests/README.md`` в корне репозитория.
