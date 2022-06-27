

Table of contents
=================
* `Installation`_
* `Getting started`_

* `Contact`_




Installation
============

The easiest way to install Met-TL is to use ``PyPI``:

.. code:: bash

  pip install mettl

Alternatively, you can checkout the repository,

.. code:: bash

  git clone https://github.com/ODU-CSM/Pub-Met-TL.git


and then install Met-TL using ``setup.py``:

.. code:: bash

  python setup.py install


Getting started
===============

1. Store known CpG methylation states of each cell into a tab-delimted file with the following columns:

* Chromosome (without chr)
* Position of the CpG site on the chromosome starting with one
* Binary methylation state of the CpG sites (0=unmethylation, 1=methylated)

Example:

.. code::

  1   3000827   1.0
  1   3001007   0.0
  1   3001018   1.0
  ...
  Y   90829839  1.0
  Y   90829899  1.0
  Y   90829918  0.0


2. Run ``dcpg_data.py`` to create the input data for DeepCpG:

.. code:: bash

  data.py
  --cpg_profiles ./cpg/cell1.tsv ./cpg/cell2.tsv ./cpg/cell3.tsv
  --dna_files ./dna/mm10
  --cpg_wlen 50
  --dna_wlen 1001
  --out_dir ./data



3. Fine-tune a pre-trained model or train your own model from scratch with ``trans_train.py``:

.. code:: bash

  trans_train.py
    ./data/c{1,3,6,7,9}_*.h5
    --val_data ./data/c{13,14,15,16,17,18,19}_*.h5
    --dna_model CnnL2h128
    --cpg_model RnnL1
    --joint_model JointL2h512
    --nb_epoch 30
    --out_dir ./model

This command uses chromosomes 1-3 for training and 10-13 for validation. ``---dna_model``, ``--cpg_model``, and ``--joint_model`` specify the architecture of the CpG, DNA, and Joint model, respectively (see manuscript for details). Training will stop after at most 30 epochs and model files will be stored in ``./model``.


4. Use ``dcpg_eval.py`` to impute methylation profiles and evaluate model performances.

.. code:: bash

  trans_eval.py
    ./data/*.h5
    --model_files ./model/model.json ./model/model_weights_val.h5
    --out_data ./eval/data.h5
    --out_report ./eval/report.tsv

This command predicts missing methylation states on all chromosomes and evaluates prediction performances using known methylation states. Predicted states will be stored in ``./eval/data.h5`` and performance metrics in ``./eval/report.tsv``.


5. Export imputed methylation profiles to HDF5 or bedGraph files:

.. code:: bash

  dcpg_eval_export.py
    ./eval/data.h5
    -o ./eval/hdf
    -f hdf






Contact
=======
* Sanjeeva Dodlapati
* sdodl001r@odu.ed.com
* https://sdodlapati.com
* `@dodlapati_reddy <https://twitter.com/dodlapati_reddy>`_
