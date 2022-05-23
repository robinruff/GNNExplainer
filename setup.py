import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gnnx-aimatlab",
    version="0.0.1",
    author="Robin Ruff",
    author_email="robin.ruff.98@googlemail.com",
    description="An implementation of the GNNExplainer by Ying et al.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robinruff/GNNExplainer",
    project_urls={
        "Bug Tracker": "https://github.com/robinruff/GNNExplainer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3"
        "Intended Audience :: Science/Research"
        "Operating System :: OS Independent"
        "Topic :: Software Development :: Libraries :: Python Modules"
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    packages=setuptools.find_packages(where="gnnx"),
    python_requires=">=3.6",
    install_requires=[
          'kgcnn',
          'tensorflow'
    ],
)
