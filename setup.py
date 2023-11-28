from setuptools import find_packages, setup

exec(open("chemcrow/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chemcrow",
    version=__version__,
    description="Accurate solution of reasoning-intensive chemical tasks, powered by LLMs.",
    author="Andres M Bran, Sam Cox, Andrew White, Philippe Schwaller",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/chemcrow-public",
    license="MIT",
    packages=find_packages(),
    package_data={'chemcrow': ['data/chem_wep.csv']},
    install_requires=[
        "ipython",
        "python-dotenv",
        "rdkit",
        "synspace",
        "openai==0.28.1",
        "gpt4all==0.3.0",
        "llama-cpp-python==0.2.13",
        "molbloom",
        "paper-qa==1.1.1",
        "google-search-results",
        "langchain>=0.0.234,<=0.0.275",
        "langchain_core==0.0.1",
        "nest_asyncio",
        "tiktoken",
        "rmrkl",
        # "paper-scraper@git+https://github.com/blackadad/paper-scraper.git",
        "streamlit",
        "rxn4chemistry",
        "duckduckgo-search",
        "wikipedia",
        "paperscraper"
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
