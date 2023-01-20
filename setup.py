from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='CLARINET',
    version='Latest',

    author='Yasmine Ahmed',
    #author_email='',
    description='CLARIfying NETworks',
    long_description='A tool that equips modelers with a fast and reliable assistant to select the most relevant knowledge about the systems being modeled using graph-based approaches and literature occurrence metadata, built by the Mechanisms and Logic of Dynamics Lab at the University of Pittsburgh',
    #license='',
    keywords='extension modeling knowledge-graph interaction',

    packages=['src'],
    include_package_data=True,

    install_requires=[
        'networkx',
        'numpy',
        'pandas',
        'community',
        'python-louvain',
        'seaborn',
        'rst2pdf',
        'tornado'
    ],
    zip_safe=False # install as directory
    )
