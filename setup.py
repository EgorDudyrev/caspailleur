import setuptools
import caspailleur

VERSION = caspailleur.__version__

def run_install(**kwargs):
    with open("README.md", "r") as fh:
        long_description = fh.read()

    install_requires = ['numpy>=1.20', 'scikit-mine>=1', 'bitarray>=2.5.1', 'tqdm']
        
    extras_require = {}

    setuptools.setup(
        name="caspailleur",
        version=VERSION,
        author="Egor Dudyrev",
        author_email="egor.dudyrev@yandex.ru",
        description="Minimalistic python package for mining many concise data representations. Part of SmartFCA project",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/EgorDudyrev/caspailleur",
        packages=setuptools.find_packages(exclude=("tests",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8',
        install_requires=install_requires,
        extras_require=extras_require,
    )


if __name__ == "__main__":
    run_install()
