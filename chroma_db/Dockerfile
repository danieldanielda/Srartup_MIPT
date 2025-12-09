FROM ghcr.io/chroma-core/chroma:latest

ENV APP_HOME=/home/chroma
RUN mkdir -p $APP_HOME
RUN mkdir -p $APP_HOME/cache
WORKDIR $APP_HOME

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y build-essential python3 python3-pip python3-venv git && \
    apt clean

ENV VIRTUAL_ENV="$APP_HOME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt