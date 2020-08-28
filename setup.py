from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='Reinforcement Learning',
      version='0.5.0',
      description='The reinforcement learning module for Python',
      long_description=readme,
      url='https://research-git.uiowa.edu/sanzabizadeh/Reinforcement-Learning',
      author='Sadjad Anzabi Zadeh',
      author_email='sadjad-anzabizadeh@uiowa.edu',
      license=license,
      packages=find_packages(exclude=('tests', 'docs')))
