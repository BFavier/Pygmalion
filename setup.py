# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygmalion",
    version="0.0.1",
    author="Benoit Favier",
    author_email="benoitfamillefavier@gmail.com",
    description="A machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas>=1.0.0",
		"numpy>=1.18",
		"torch>=1.2",
		"python>=3.6"
    ],                                             
    url="https://github.com/BFavier/Pygmalion",  
    packages=setuptools.find_packages(),
    classifiers=(                                 # Classifiers help people find your 
        "Programming Language :: Python :: 3",    # projects. See all possible classifiers 
        "License :: OSI Approved :: MIT License", # in https://pypi.org/classifiers/
        "Operating System :: OS Independent",
		"Environment :: GPU :: NVIDIA CUDA"
    ),
)
