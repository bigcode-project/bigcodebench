from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

with open("requirements-wildcodebench.txt") as reqs_file:
    wildcodebench_requirements = reqs_file.read().split("\n")

setup(
    name="wildcodebench",
    py_modules=["wildcodebench"],
    version="0.1",
    python_requires='>=3.8',
    description="WildCodeBench: A Rigorous Benchmark for Code Generation with Realistic Constraints in the Wild",
    long_description=readme,
    license="Apache 2.0",
    packages=find_packages() ,
    install_requires=requirements,
    extras_require={"wildcodebench": wildcodebench_requirements},
    entry_points={
        "console_scripts": [
            "wildcode.evaluate = wildcode.evaluate:main",
            "wildcode.sanitize = wildcode.sanitize:main",
            "wildcode.syncheck = wildcode.syncheck:main",
            "wildcode.legacy_sanitize = wildcode.legacy_sanitize:main",
            "wildcode.generate = wildcode.generate:main",
            "wildcode.inspect = wildcode.inspect:main",
        ]
    },
)