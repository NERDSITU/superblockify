Getting Started
***************

To set up an environment, the simplest way is to use `conda` or `mamba` (exchange
`conda` with `mamba` in any of the commands) and the `environment.yml` file. This
will create an environment called `sb_env` with all the necessary dependencies.

.. code-block:: bash
    conda env create --file=environment.yml


Alternatively, a version-less setup can be done by resolving the dependencies
separately.

.. code-block:: bash
    conda create -n sb_env -c conda-forge python=3.10 --file requirements.txt
    conda activate sb_env

To inspect the changes, run
`conda env export | grep -v "^prefix: " > environment.yml`
and see the differences in versions using `git diff environment.yml`.

Usage
-----

The `superblockify` package works out of the box, meaning no further downloads are
necessary. Maps are downloaded from the OpenStreetMap API and population data is
downloaded from the `GHSL-POP 2023 <https://ghsl.jrc.ec.europa.eu/ghs_pop2023.php>`_
dataset. Only tiles needed are being cached in the `data/ghsl` folder.

As example we will try to partition the french city of Lyon with the
`ResidentialPartitioner` class. Afterwards we will save the results to a geopackage
file that can easily be opened and edited in QGIS.

.. code-block:: python
    import superblockify as sb
    part = sb.ResidentialPartitioner(
        name="Lyon_test", city_name="Lyon", search_str="Lyon, France"
    )
    part.run(calculate_metrics=False, make_plots=True)
    sb.save_to_gpkg(part, save_path=None)

This will download the map of Lyon, store it in the `data/graphs` folder for
use later, partition the city and save the results to a geopackage file in the
`data/results/Lyon_test` folder. In the same folder, a `Lyon_test.gpkg` file
will be created that can be opened in QGIS.
If you want to also calculate the metrics, you can set `calculate_metrics=True`.

To learn more about the inner workings and background of the package, please
see the next :ref:`reference pages <guide>`.


Testing
-------

The tests are specified using the `pytest` signature in the `tests/` folder and
can be run using a test runner of choice.
