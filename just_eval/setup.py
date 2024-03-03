import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

about = {}
with open("just_eval/_version.py") as f:
    exec(f.read(), about)
os.environ["PBR_VERSION"] = about["__version__"]

setuptools.setup(
    name="just_eval", 
    version=about["__version__"],
    author='Bill Yuchen Lin',
    author_email='yuchenl@allenai.org',
    description="A simple and easy tool for evaluate LLMs using GPT APIs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Re-Align/just_eval",
    py_modules=[],
    packages=setuptools.find_packages(),
    install_requires=['argparse',
                      'tqdm',
                      'openai',
                      'numpy'
                    ],
    entry_points = {
        'console_scripts': ['just_eval=just_eval.evaluate:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6', 
)