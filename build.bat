del dist\*.tar.gz
del dist\*.whl
python setup.py sdist bdist_wheel
RMDIR /S /Q .\build
RMDIR /S /Q .\pygmalion.egg-info
pause