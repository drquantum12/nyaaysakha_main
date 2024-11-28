FROM python:3.11-slim

WORKDIR /main

COPY ./requirements.txt /main/requirements.txt


RUN pip install --no-cache-dir -r /main/requirements.txt

COPY ./app /main/app

ENV config_path=app/config.ini

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]


