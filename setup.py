import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="robo_engine",  
    version="0.0.1",
    author="michael_yuan", 
    author_email="ycb24@mails.tsinghua.edu.cn",  
    description="robo_engine: a plug-and-play visual robot data augmentation toolkit",
    long_description=long_description,  
    url="https://github.com/michaelyuancb/robo_engine",  
    packages=setuptools.find_packages(), 
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)