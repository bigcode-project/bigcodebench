# Better use newer Python as generated code can use new features
FROM python:3.10-slim

# install git
RUN apt-get update && apt-get install -y git

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /openeval

RUN cd /openeval && pip install .

# Pre-install the dataset
RUN python3 -c "from openeval.data import get_open_eval_plus;"

WORKDIR /app

ENTRYPOINT ["python3", "-m", "openeval.evaluate"]
