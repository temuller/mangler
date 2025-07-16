import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open("mangler/_version.py") as version_file:
    for line in version_file:
        if "__version__" in line:
            __version__ = line.split()[-1].replace('"', "")

setuptools.setup(
    name="mangler",
    version=__version__,
    author="Tomás Enrique Müller-Bravo",
    author_email="t.e.muller-bravo@ice.csic.es",
    license="MIT",
    description="Code for mangling type Ia SED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/mangler",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    include_package_data=True,
)
