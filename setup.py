import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gwot", # Replace with your own username
    version="0.0.10",
    author="Stephen Zhang",
    author_email="syz@math.ubc.ca",
    description="gWOT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsteve/gwot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
