from setuptools import setup, find_packages

setup(
    name='edgecam',
    version='0.1.0',
    author='Seunghyeon Kim',
    author_email='kimtmdgus22@gmail.com',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['edgecam.*']),
    install_requires=[],
    classifiers=[],
    python_requires='>=3.8',
)
