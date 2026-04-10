.. _local-batch:

Локальный запуск batch-обработки
================================

Если нужно обработать директорию локально без web API:

.. code-block:: bash

   source venv/bin/activate
   python process.py

По умолчанию ожидается директория:

.. code-block:: text

   ./input_video/

Результаты локального режима сохраняются в:

* ``./output_audio``
* ``./output_data``
