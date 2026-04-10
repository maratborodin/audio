.. _output-format:

Формат результата
=================

Итоговый JSON содержит:

* ``transcription``
* ``diarization``
* для каждого спикера:

  * ``segments``
  * ``voice_metrics``
  * ``voice_description``

Результат сохраняется в исходный бакет под ключом:

.. code-block:: text

   output_data/<имя_исходного_видео>.json

Например:

.. code-block:: text

   input_video/interview.webm -> output_data/interview.json
