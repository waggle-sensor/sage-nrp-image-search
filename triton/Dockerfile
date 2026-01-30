FROM nvcr.io/nvidia/tritonserver:24.06-pyt-python-py3

RUN apt-get update \
  && apt-get install -y \
  wget \
  curl \
  libgl1 \
  libglib2.0-0 \
  git

WORKDIR /app

COPY requirements.txt .
COPY torch_requirements.txt .

RUN pip install --no-cache-dir -r torch_requirements.txt
RUN pip install --no-cache-dir  --no-build-isolation -r requirements.txt

COPY models models
COPY HyperParameters.py HyperParameters.py
COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
