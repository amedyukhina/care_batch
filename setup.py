from setuptools import setup
import care_batch

setup(
    name='care_batch',
    version=care_batch.__version__,
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['care_batch'
              ],
    license='Apache License 2.0',
    include_package_data=True,

    test_suite='care_batch.tests',

    install_requires=[
        'ipykernel',
        'scipy',
        'ddt',
        'pytest',
        'tqdm',
        'scikit-image',
        'pandas',
        'seaborn',
        'jupyter',
        'keras>=2.2.5,<2.3.0',
        'tensorflow-gpu>=1.15,<2',
        'csbdeep',
        'am_utils @ git+https://github.com/amedyukhina/am_utils.git'
    ],
)