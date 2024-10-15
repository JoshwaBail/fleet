from setuptools import setup, find_packages

setup(
    name="fleet",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the fleet library",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "openai",
        "anthropic",
        "groq"
    ],
)