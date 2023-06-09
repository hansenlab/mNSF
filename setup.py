from setuptools import setup, find_packages

setup(
    name='mNSF',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scanpy',
        'anndata',
        'tensorflow',
        'tensorflow-probability',
        'dill',
        'squidpy'
    ],
    author='Yingxin Lin',
)