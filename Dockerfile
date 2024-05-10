# Better use newer Python as generated code can use new features
FROM python:3.10-slim

# install git
RUN apt-get update && apt-get install -y git && apt-get -y install g++ && apt-get install python3-tk

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /wildcode

RUN cd /wildcode && pip install . && pip install -r https://raw.githubusercontent.com/bigcode-project/open-eval/main/requirements.txt

# Pre-install the dataset
RUN python3 -c "from wildcode.data import get_wild_code_bench;"

WORKDIR /wildcode

ENTRYPOINT ["python3", "-m", "wildcode.evaluate"]
