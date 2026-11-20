from setuptools import setup

setup(
    name='ccn',
    version='0.0.1',
    packages=['ccn'],
    install_requires=[
        'matplotlib>=3.5',
        'networkx>=2.6',
        'numpy>=1.21',
        'pytest',
        'torch>=1.8',
        'scipy'
    ],
)