from setuptools import setup, find_packages
from typing import List

hyphon_e_dot = "-e ."

def get_requirements(filepath: str) -> List[str]:
    requirements = []
    
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace('\n',"") for i in requirements]
        
        if hyphon_e_dot in requirements:
            requirements.remove(hyphon_e_dot)
        
setup(name='ML_pipeline',
      version='0.1',
      description='Ml_pipeline',
      author='Bhanu Pratap Singh',
      author_email='bhanupsingh484@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements("req.txt")
)