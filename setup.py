from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'A turbomachinery design package.'
LONG_DESCRIPTION = 'A tool-set package for the design of turbomachinery components, for use in liquid rocket engine design.'

# Setting up
setup(
    name="turboRocket",
    version=VERSION,
    author="Elias Aoubala (Elias Aoubala)",
    author_email="<elias.aoubala@outlook.com>",
    license="MIT",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy',
        'cantera',
        'matplotlib',
        'scipy',
        'pandas',
        'yaml'
        ],
    
    keywords=['python', 'rocket propulsion', 'LRE', 'turbomachinery'],
    
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)