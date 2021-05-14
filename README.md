# DarkReactor

DarkReactor is a supplementary module to the
[DarkChem](https://github.com/pnnl/darkchem) package developed at the
Pacific Northwest National Laboratory.

## Usage

DarkReactor is a suite of Python modules enabling the batch analysis
of molecular reactions. While the code has been written to be
compatible with [DarkChem](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b02348),
its functionalities can easily be adapted to other frameworks.

To install and use DarkReactor in its original implementation
with DarkChem:

1. Follow the
   [DarkChem installation guide](https://github.com/pnnl/darkchem) to
   install DarkChem on your system via conda.
2. Use pip to install the code directly from the repository:
```bash
# install from an existing copy of the repo
cd /path/to/darkreactor
pip install .

# clone/install
git clone https://github.com/pnnl/darkreactor.git
pip install darkreactor/

# direct
pip install git+https://github.com/pnnl/darkreactor
```
3. DarkReactor is now ready to use!
  - Note: To run DarkReactor on the included data file, you must
    (1) format data and artificially generate reaction products
    using `generate_data.py`, and then (2) calculate reaction vectors
    and assess predictions using `run_darkreactor.py`.

# Disclaimer

This material was prepared as an account of work sponsored by an
agency of the United States Government. Neither the United States
Government nor the United States Department of Energy, nor Battelle,
nor any of their employees, nor any jurisdiction or organization
that has cooperated in the development of these materials, makes any
warranty, express or implied, or assumes any legal liability or
responsibility for the accuracy, completeness, or usefulness or any
information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or
service by trade name, trademark, manufacturer, or otherwise does
not necessarily constitute or imply its endorsement, recommendation,
or favoring by the United States Government or any agency thereof,
or Battelle Memorial Institute. The views and opinions of authors
expressed herein do not necessarily state or reflect those of the
United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the
UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL0 1830
