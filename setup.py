import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anagen-hansonlu", # Replace with your own username
    version="0.0.1",
    author="Hanson Lu",
    author_email="hansonlu@uchicago.edu",
    description="Anaphor generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hansonlu/anagen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
