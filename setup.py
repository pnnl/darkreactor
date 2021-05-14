from setuptools import setup, find_packages
from darkreactor import __version__

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = [x for x in f.read().splitlines() if not x.startswith("#")]

pkgs = find_packages(exclude=('data', 'resources'))

# Note: the _program variable is set in __init__.py.
# it determines the name of the package/final command line tool.

setup(
    name='darkreactor',
    version=__version__,
    description=('Software package for generating latent space'
                 'reaction vectors from molecular data.'),
    url='http://github.com/pnnl/darkreactor/',
    author='@christinehc',
    author_email='christine.chang@pnnl.gov',
    license=license,
    packages=pkgs,
    install_requires=required,
    zip_safe=False)
