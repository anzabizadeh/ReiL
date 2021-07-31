# type: ignore

import setuptools
import versioneer

with open('README.rst') as f:
    long_description = f.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name='ReiL',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A Reinforcement Learning Module for Python',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://research-git.uiowa.edu/sanzabizadeh/Reinforcement-Learning',
    author='Sadjad Anzabi Zadeh',
    author_email='sadjad-anzabizadeh@uiowa.edu',
    license='MIT License',
    packages=setuptools.find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
