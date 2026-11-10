from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent
requirements = [
    line.strip()
    for line in (root / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
setup(
    name="foresight-navsim",
    version="0.1.0",
    description="ForeSight: future-conditioned autonomous driving planning",
    url="https://github.com/LogosIntelligenceLab/ForeSight",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10,<3.12",
    install_requires=requirements,
    license="MIT",
    license_files=("LICENSE.md", "LICENSES/*.txt"),
)
