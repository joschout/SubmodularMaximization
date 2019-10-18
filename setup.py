from setuptools import setup

with open("README.md", "r") as fh:

    long_description = fh.read()

setup(
    name='Unconstrained Submodular Maximization',
    version='0.0.1',
    packages=[''],
    package_dir={'': 'submodmax'},
    url='https://github.com/joschout/SubmodularMaximization',
    license='Apache 2.0',
    author='Jonas Schouterden',
    author_email='',
    description='Unconstrained submodular maximization optimization algorithms',
    long_description=long_description,
    install_requires=['numpy']
)
