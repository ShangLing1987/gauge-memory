from setuptools import setup, find_packages

setup(
    name="gauge-memory",
    version="0.1.0",
    description="Physics-inspired hierarchical memory with contradiction detection and Langevin forgetting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="GaugeMemory contributors",
    url="https://github.com/your-org/gauge-memory",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
