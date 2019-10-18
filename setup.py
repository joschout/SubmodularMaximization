import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name='submodmax',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/joschout/SubmodularMaximization',
    license='Apache 2.0',
    author='Jonas Schouterden',
    author_email='',
    description='Unconstrained submodular maximization optimization algorithms',
    long_description=long_description,
    install_requires=['numpy']
)
