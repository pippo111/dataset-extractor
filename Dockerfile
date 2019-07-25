FROM jupyter/tensorflow-notebook

WORKDIR /usr/src/app

# RUN apt-get update
# RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y python3-tk

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "./main.py" ]
