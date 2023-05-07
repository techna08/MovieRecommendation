FROM python:3.9
COPY flask_app.py /
COPY provenance.py /
COPY MLModels/*.py /MLModels/

COPY MLModels/model_baseline.bin /

COPY requirements.txt /
# WORKDIR /
# ADD . /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8081
ENTRYPOINT [ "python3" ]
CMD [ "flask_app.py", "model_baseline.bin" ]