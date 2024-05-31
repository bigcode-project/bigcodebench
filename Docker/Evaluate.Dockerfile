# Better use newer Python as generated code can use new features
FROM python:3.10-slim

# install git, g++ and python3-tk
RUN apt-get update && apt-get install -y git g++ python3-tk zip unzip procps r-base

# upgrade to latest pip
RUN pip install --upgrade pip

# Add a new user "bigcodebenchuser"
RUN adduser --disabled-password --gecos "" bigcodebenchuser

# Acquire benchmark code to local
RUN git clone https://github.com/bigcode-project/code-eval.git /bigcodebench

RUN cd /bigcodebench && pip install . && pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench-annotation/main/requirements.txt

# Pre-install the dataset
RUN python3 -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench()"

RUN chown -R bigcodebenchuser:bigcodebenchuser /bigcodebench
USER bigcodebenchuser

WORKDIR /bigcodebench

ENTRYPOINT ["python3", "-m", "bigcodebench.evaluate"]