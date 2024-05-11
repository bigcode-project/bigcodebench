# Better use newer Python as generated code can use new features
FROM python:3.10

# install git, g++ and python3-tk
RUN apt-get update && apt-get install -y git g++ python3-tk

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /wildcode

RUN cd /wildcode && pip install . && pip install -r https://raw.githubusercontent.com/bigcode-project/wildcodebench-annotation/main/requirements.txt

# Pre-install the dataset
RUN python3 -c "from wildcode.data import get_wildcodebench; get_wildcodebench()"

WORKDIR /wildcode
