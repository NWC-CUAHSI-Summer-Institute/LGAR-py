from setuptools import setup, find_packages

setup(
    name='dpLGAR',  # Name of your package
    version='0.1',  # Version number
    description='LGAR in torch',  # Short description
    url='https://github.com/NWC-CUAHSI-Summer-Institute/dpLGAR/tree/main',  # URL to the github repo
    author='Your Name',  # Your name
    author_email='tkb5476@psu.edu',  # Your email
    license='MIT',  # License type
    packages=find_packages(),  # Automatically find all packages
    install_requires=[],  # List of dependencies (as strings)
    zip_safe=False  # Not necessary, set to False if unsure
)
