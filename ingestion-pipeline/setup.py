"""Setup script for Regulatory Chat Bot Clean Ingestion."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="regulatory-chatbot-ingestion",
    version="0.1.0",
    author="Datafactz",
    author_email="info@datafactz.com",
    description="Clean ingestion pipeline for regulatory document processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datafactz/regulatory-chatbot",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "regulatory-pipeline=scripts.pipeline_cli:main",
            "regulatory-process=scripts.process_documents:main",
            "regulatory-monitor=scripts.monitor_pipeline:main",
            "regulatory-reset=scripts.reset_database_standalone:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
)