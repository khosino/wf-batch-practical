FROM python:3.8
RUN mkdir /code　
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY src/main.py /code/
CMD ["python", "main.py"]