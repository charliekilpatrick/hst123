.. _installation:

Installation
==============

Python and environment
------------------------

**hst123** requires **Python 3.12** only (see ``requires-python`` in ``pyproject.toml``: ``>=3.12,<3.13``). The stack is validated with NumPy 1.x, Astropy 5.x, SciPy 1.x, and DrizzlePac; use the provided Conda environment when possible so binary wheels match.

Conda (recommended)
~~~~~~~~~~~~~~~~~~~

From the repository root:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate hst123
   pip install drizzlepac
   pip install -e .

This installs the ``hst123`` and ``hst123-install-dolphot`` console scripts.

``pip install`` alone (without Conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a **venv with Python 3.12**, then ``pip install drizzlepac`` and ``pip install -e .`` from the repo. If NumPy/Astropy must build from source, prefer installing those from Conda first or match the pins in ``pyproject.toml``.

STScI WCS (``stwcs``)
~~~~~~~~~~~~~~~~~~~~~

WCS utilities used for alignment (``updatewcs``, ``HSTWCS``, ``altwcs``) are **bundled** under ``hst123.utils.stwcs`` (see ``STWCS_VENDOR.txt``). The package on PyPI named ``stwcs`` is **not** an hst123 dependency; DrizzlePac may install it for its own use. Application code should use ``hst123.utils.stsci_wcs``.

DOLPHOT (optional)
~~~~~~~~~~~~~~~~~~~

DOLPHOT is **not** bundled. For ``--run-dolphot`` / ``--scrape-dolphot``, install DOLPHOT with the helper after activating your environment:

.. code-block:: bash

   conda activate hst123
   hst123-install-dolphot

The helper downloads DOLPHOT 3.1 sources and PSF tables under ``$CONDA_PREFIX/opt/hst123-dolphot``, runs ``make``, and symlinks ``dolphot``, ``calcsky``, and related tools into ``$CONDA_PREFIX/bin``. Masking (ACS/WFC3/WFPC2) and ``splitgroups`` use **Python implementations** by default (see ``hst123.utils.dolphot_mask``, ``hst123.utils.dolphot_splitgroups``); set ``HST123_DOLPHOT_MASK_EXTERNAL=1`` or ``HST123_DOLPHOT_SPLITGROUPS_EXTERNAL=1`` to force C binaries.

Without Conda, pass ``--dolphot-dir /path/to/dolphot`` to the installer or add DOLPHOT to ``PATH`` manually. See ``hst123-install-dolphot --help`` and the `DOLPHOT website <http://americano.dolphinsim.com/dolphot/>`_.

Documentation build
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

Output is under ``docs/build/html/``. The project also publishes docs to `GitHub Pages <https://charliekilpatrick.github.io/hst123/>`_ via CI.
