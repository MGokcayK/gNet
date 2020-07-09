import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gNet", 
    version="0.1.1",
    author="Mehmet Gökçay Kabataş",
    author_email="mgokcaykdev@gmail.com",
    description="A mini Deep Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MGokcayK/gNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)