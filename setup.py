# Copyright 2021 c00k1ez (https://github.com/c00k1ez). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import setuptools


def version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "src/plain_transformers", "__init__.py")
    with open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#"):
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        if ln.startswith("http"):
            continue
        if ln:
            reqs.append(ln)
    return reqs


setuptools.setup(
    name="plain-transformers",
    version=version(),
    author="c00k1ez",
    author_email="c00k1ez.th13f@gmail.com",
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
    packages=setuptools.find_packages(where="src", exclude=["tests/", "tests/*"]),
    python_requires=">=3.6",
    install_requires=_load_requirements(os.path.dirname(__file__)),
    include_package_data=True,
)
