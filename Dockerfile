FROM python:3.9.19-slim-bookworm
COPY . /app
WORKDIR /app
RUN  pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python main.py