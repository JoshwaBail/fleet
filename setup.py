from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fleet-agents",
    version="0.2.0",
    author="Fleet Contributors",
    author_email="contact@fleetagents.dev",
    description="A lightweight, nautical-themed LLM agent builder for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fleet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "termcolor>=2.0.0",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "examples": [
            "python-dotenv>=1.0.0",
        ],
    },
    keywords="llm ai agents chatbot openai anthropic claude gpt multi-agent tools function-calling",
    project_urls={
        "Documentation": "https://github.com/yourusername/fleet#readme",
        "Source": "https://github.com/yourusername/fleet",
        "Bug Reports": "https://github.com/yourusername/fleet/issues",
    },
)
