import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _load_requirements(
        path_dir: str,
        file_name: str = 'requirements.txt',
        comment_char: str = '#'
):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith('http'):
            continue
        if ln:
            reqs.append(ln)
    return reqs


setuptools.setup(
    name="plain-tranformers",
    version="0.0.1",
    author="c00k1ez",
    author_email="egorplotnikov18@gmail.com",
    description="one more transformers lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/c00k1ez/plain_transformers",
    project_urls={
        "Bug Tracker": "https://github.com/c00k1ez/plain_transformers/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=['tests/', 'tests/*']),
    python_requires=">=3.6",
    install_requires=_load_requirements(os.path.dirname(__file__))
)
