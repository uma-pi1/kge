from setuptools import setup

setup(
    name="libkge",
    version="0.1",
    description="A knowledge graph embedding library",
    url="https://github.com/uma-pi1/kge",
    author="Universität Mannheim",
    author_email="rgemulla@uni-mannheim.de",
    packages=["kge"],
    install_requires=[
        "numpy>=1.19.*",
        "torch>=1.7.1",
        "pyyaml",
        "pandas",
        "argparse",
        "path",
        # please check correct behaviour when updating ax platform version!!
        "ax-platform>=0.1.19", "botorch>=0.4.0", "gpytorch>=1.4.2",
        "sqlalchemy",
        "torchviz",
        # LibKGE uses numba typed-dicts which is part of the experimental numba API
        # see http://numba.pydata.org/numba-doc/0.50.1/reference/pysupported.html
        "numba>=0.50.*",
        "hpbandster",
        "ConfigSpace",
        "pytest-shutil",
        "igraph",
    ],
    python_requires=">=3.7",
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
