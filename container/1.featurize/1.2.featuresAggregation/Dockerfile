FROM python:3.8
RUN mkdir /code　
COPY requirements.txt /code/
COPY src/main.py /code/

WORKDIR /code
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "./main.py"]
CMD ["0"]