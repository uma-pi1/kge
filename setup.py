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
        "numpy==2.4.3",
        "torch==2.10.0",
        "pyyaml",
        "pandas==3.0.1",
        "argparse",
        "path==17.1.1",
        "ax-platform==1.2.4", "botorch==0.17.2", "gpytorch==1.15.2",
        "sqlalchemy<2.0.0",
        "torchviz==0.0.3",
        # LibKGE uses numba typed-dicts which is part of the experimental numba API
        # see http://numba.pydata.org/numba-doc/0.50.1/reference/pysupported.html
        "numba==0.64.0",
        "hpbandster",
        "ConfigSpace",
        "pytest-shutil",
        "igraph",
    ],
    python_requires=">=3.7",
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
