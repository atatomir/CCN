from setuptools import setup

setup(
    name='ccn',
    version='0.0.1',
    packages=['ccn'],
    install_requires=[
        'matplotlib==3.5.1',
        'networkx==2.6.3',
        'numpy==1.21.2',
        'pytest==7.0.1',
        'torch==1.8.0',
        'scipy'
    ],
)