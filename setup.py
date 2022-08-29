from setuptools import setup

def readme():
    with open('README_clarinet') as f:
        return f.read()

setup(
    name='CLARINET',
    version='1.0',
    
    author='Yasmine Ahmed',
    #author_email='',
    description='CLARINET',
    long_description='Verifying Interactions Of Likely Importance to the Network, built by the Mechanisms and Logic of Dynamics Lab at the University of Pittsburgh',
    #license='',
    keywords='dynamic system boolean logical qualitative modeling simulation',

    packages=['CLARINET'],
    include_package_data=True,

    install_requires=[
        'networkx',
        'numpy',
        'pandas',
        'community',
        'seaborn',
        'rst2pdf',
        'tornado==4.5.3' # to not interfere with jupyter
    ],
    zip_safe=False # install as directory
    )