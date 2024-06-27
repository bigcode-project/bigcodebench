# Better use newer Python as generated code can use new features
FROM python:3.10-slim

# install git, g++ and python3-tk
RUN apt-get update && apt-get install -y git g++ python3-tk zip unzip procps r-base

# upgrade to latest pip
RUN pip install --upgrade pip

# Add a new user "bigcodebenchuser"
RUN adduser --disabled-password --gecos "" bigcodebenchuser

RUN rm -rf /bigcodebench

# Acquire benchmark code to local
ADD "https://api.github.com/repos/bigcode-project/bigcodebench/commits?per_page=1" latest_commit
RUN git clone https://github.com/bigcode-project/bigcodebench.git /bigcodebench

RUN cd /bigcodebench && pip install .

# Pre-install the dataset
RUN python3 -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench()"

RUN pip install -I --timeout 2000 -r https://github.com/bigcode-project/bigcodebench-annotation/releases/download/v0.1.0/requirements.txt

WORKDIR /app

RUN chown -R bigcodebenchuser:bigcodebenchuser /app

RUN chmod -R 777 /app

USER bigcodebenchuser

ENTRYPOINT ["python3", "-m", "bigcodebench.evaluate"]