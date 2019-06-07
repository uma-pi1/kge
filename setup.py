from setuptools import setup

setup(
    name="kge",
    version="0.1",
    description="A knowledge graph embedding library",
    url="https://github.com/rufex2001/kge",
    author="Universität Mannheim",
    author_email="rgemulla@uni-mannheim.de",
    packages=["kge"],
    install_requires=[
        "torch>=1.0.1",
        "pyyaml",
        "pandas",
        "argparse",
        "path.py",
        "ax-platform",
        "sqlalchemy",
    ],
    zip_safe=False,
)
