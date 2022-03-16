from setuptools import setup, find_packages
import os 
import shutil

# run 1: python setup.py build
# run 2: python setup.py install
            
setup(
    name='pyStruct',
    version="0.0.6",
    license='None',
    author='yujou',
    author_email='yjouwang@mit.edu',
    description='Structure prediction',
    packages=find_packages(),
        
)


# Clean the file after setup
remove_list = ['build', 'dist', 'pyStruct.egg-info', '__pycache__']
print(f'Finish installing. Clean up files...')
for remove in remove_list:
    try:
        shutil.rmtree(remove)
    except:
        print(f"Fail remove {remove}")