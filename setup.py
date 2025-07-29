"""Setup configuration for vision_evals package."""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='vision-evals',
    version='0.1.0',
    author='Qasim Wani',
    author_email='qasim31wani@gmail.com',
    description='A PyPI package for object detection using advanced vision models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/vision-evals',
    project_urls={
        'Bug Tracker': 'https://github.com/yourusername/vision-evals/issues',
        'Documentation': 'https://github.com/yourusername/vision-evals#readme',
        'Source Code': 'https://github.com/yourusername/vision-evals',
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['tests*', 'docs*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.990',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision-evals=run_cli:main',
        ],
    },
    include_package_data=True,
    keywords='computer vision, object detection, AI, machine learning, OpenAI, Gemini',
)