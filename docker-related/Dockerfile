#FROM python:3.13.0a2-bookworm
FROM python:3.10.13-bullseye
#FROM 3.10.13-alpine3.19
WORKDIR /usr/src/app
COPY ./proj/torch-ecpg/requirements.txt .
COPY ./proj/torch-ecpg/setup.py .
COPY ./proj/torch-ecpg/README.md .
COPY tecpg .
RUN pip config --user set global.progress_bar off
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY ./proj .
RUN pip install .

COPY . .

ENV PYTHONPATH "/usr/src/app/torch-ecpg:${PYTHONPATH}"

CMD [ "python", "./tecpg" ]


