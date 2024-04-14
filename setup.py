from setuptools import setup, find_packages

# Load text for description and license
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="pbeis",
    version="0.1.0",
    description="PyBaMM EIS",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/rish31415/pybamm-eis",
    packages=find_packages(include=("pbeis")),
    author="Rishit Dhoot & Robert Timms",
    author_email="timms@maths.ox.ac.uk",
    license="LICENSE",
    install_requires=["pybamm[all]==24.1", "matplotlib"],
)
