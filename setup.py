import subprocess
from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as file:
        return(file.read())

setup(name='emission_line_fitting',
      version='0.0',
      description='Spectral fitting of emission lines',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/joriswitstok/emission-line-fitting',
      author='Joris Witstok',
      author_email='jnw30@cam.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'matplotlib==3.5.3',
        'numpy>=1.23.2',
        'scipy>=1.9.0',
        'astropy>=5.1',
        'spectres>=2.2.0',
        'corner',
        'seaborn',
        'pymultinest',
      ],
      python_requires='>=3.8',
      #entry_points={
      #  'console_scripts': [
      #      'nirspecxf      = nirspecxf.samisimsig_script:main',
      #      ],
      #},
     )
