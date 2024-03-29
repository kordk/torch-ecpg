import os

from setuptools import find_packages, setup

dir_path = os.path.dirname(__file__)

readme_file = 'README.md'
readme_path = os.path.join(dir_path, readme_file)
with open(readme_path, 'r') as file:
    long_description = file.read()

requirements_file = 'requirements.txt'
requirements_path = os.path.join(dir_path, requirements_file)
with open(requirements_path, 'r') as file:
    requirements = [line.removesuffix('\n') for line in file.readlines()]

setup(
    name='tecpg',
    version='0.0.1',  # See tecpg/tecpg/__init__.py
    description='Python eCpG mapper with CLI using pytorch',
    long_description=long_description,  # See tecpg/README.md
    python_requires='>=3.10',
    packages=find_packages(),
    entry_points={'console_scripts': ['tecpg = tecpg.__main__:main']},
    install_requires=requirements,  # See tecpg/requirements.txt
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: Console',
        'Typing :: Typed',
    ],
)
