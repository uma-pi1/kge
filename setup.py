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
        "torch>=1.3.1",
        "pyyaml",
        "pandas",
        "argparse",
        "path.py",
        "ax-platform>=0.1.6",
        "sqlalchemy",
        "torchviz",
        "dataclasses"
    ],
    zip_safe=False,
)
