# -*- coding: utf-8 -*-
import setuptools

with open("pygmalion/_info.py", "r") as fh:
    lines = fh.readlines()
    key_values = [line.rstrip().replace("\"", "").split(" = ")
                  for line in lines]
    infos = {key: value for key, value in key_values}

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = fh.read().split()

setuptools.setup(
    name="pygmalion",
    version=infos["__version__"],
    author=infos["__author__"],
    author_email="benoitfamillefavier@gmail.com",
    description="A machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    python_requires='>=3.8',
    url="https://github.com/BFavier/Pygmalion",
    packages=setuptools.find_packages(),
    classifiers=(                                 # Classifiers help people find your 
        "Programming Language :: Python :: 3",    # projects. See all possible classifiers 
        "License :: OSI Approved :: MIT License", # in https://pypi.org/classifiers/
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA"
    ),
)
