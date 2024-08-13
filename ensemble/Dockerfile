FROM python:3.6
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get install g++
RUN pip install pybind11==2.2.4
RUN pip install --no-cache-dir -r requirements.txt
COPY . /code/
