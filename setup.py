from setuptools import setup, find_packages


CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Mathematics'
]

setup(
    name='pbcs',
    version='0.2.0',
    description='Projected BCS code for seniority-conserving Hamiltonians',
    author='Guo P. Chen',
    author_email='peizhi.chen@gmail.com',
    license='MIT',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=['numpy', 'pyscf>=2.0.1', 'cyipopt>=1.1.0']
)
