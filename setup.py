from setuptools import find_packages, setup

setup(
    name = 'tsatools',
    packages = find_packages(include=['tsatools']),
    version = '0.2.0',
    description = 'Library of commmonly used functions to manipulate and analyse time series data.',
    author = 'Ross Duncan',
    install_requires = ['pandas',
                        'statsmodels',
                        'matplotlib',
                        'scikit-learn',
                        'seaborn'],
    tests_require = ['pytest-runner'],
    test_suite = 'tests',
)