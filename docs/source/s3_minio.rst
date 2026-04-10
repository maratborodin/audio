.. _s3-minio:

Особенности работы с S3 / MinIO
===============================

Поддерживаются форматы:

* ``s3://bucket/prefix``
* ``s3://127.0.0.1:9000/bucket/prefix``
* ``http://127.0.0.1:9000/bucket/prefix``
* ``https://bucket.s3.amazonaws.com/prefix``

Для MinIO обычно используется API порт ``9000``, а не web-console порт ``9001``.

Пример проверки содержимого бакета через ``mc``:

.. code-block:: bash

   ./mc ls local/test-bucket/input_video

Пример проверки результатов:

.. code-block:: bash

   ./mc ls local/test-bucket/output_data
