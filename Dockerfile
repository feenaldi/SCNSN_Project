# build image on python
FROM docker.io/python:3.11-slim

# set working directory
WORKDIR /home/docker

# copy the requirements
COPY requirements.txt .

# install libraries
RUN pip3 install --no-cache-dir -r requirements.txt

# allows to delete ouput files
USER developer 

# ignores warnings form Keras/TesnorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2

# executes the Higgs.py script ignoring the warnings
CMD ["python", "-W", "ignore", "Higgs.py"]