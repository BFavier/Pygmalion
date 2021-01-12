from setuptools import setup,find_packages
import pygmalion as ml
setup(name="pygmalion", version=ml.__version__, packages=find_packages(), 
install_requires=['torch>=1.2.0', 'pandas>=0.25.1', 'scipy>=1.3.1', 'numpy>=1.16.5', 'matplotlib>=3.1.1'],
python_requires='>=3.6')