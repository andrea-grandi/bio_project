import os
from setuptools import find_packages, setup


# Utility function to read the README file.
# Used for the long_description.
def read(file_name):
    return open(os.path.join(os.path.dirname(__file__),
                             file_name)).read()


setup(
    name="bio_project",
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=["tests", "*visualization"]),
    version="0.1.0",
    author="andreagrandi",
    author_email="andrea.grandi@tiscali.it",
    description="artificial intelligence in bioinformatic project",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    python_requires='>=3.6, <4',
)
