from setuptools import find_packages, setup
from typing import List
HYPHON_E_DOT = '-e .'

def get_requirements(filepath:str, ) -> List[str]:
    
    requirements = []
    
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]
        
        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)
    
    
    setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Machine Learning Pipeline Project',
    author='BR Mehta Technologies',
    author_email = 'sambhavm22@gmail.com',
    license='',
    install_requires=requirements
)
