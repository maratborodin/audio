.. _task-database:

База данных задач
=================

По умолчанию используется:

.. code-block:: text

   sqlite:///./tasks.db

В таблице ``tasks`` хранятся:

* ``id``
* ``status``
* ``s3_url``
* ``prefix``
* ``progress``
* ``total``
* ``current_file``
* ``result``
* ``error``
* ``logs``
* ``created_at``
* ``updated_at``
