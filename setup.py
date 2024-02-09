from setuptools import setup, find_packages

setup(
    name="graphreadability",
    version="0.1.0",
    author="Philip Mathieu",
    author_email="mathieu.p@northeastern.edu",
    description="A Python module for applying readability metrics graph and network visualizations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PhilipMathieu/graphreadability",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "networkx",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: All Rights Reserved",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    license="All Rights Reserved",
)
