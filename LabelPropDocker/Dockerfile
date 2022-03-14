FROM nvcr.io/nvidia/pytorch:22.02-py3
COPY $PLEX_PATH /workspace 
COPY requirements.txt /workspace/requirements.txt
RUN apt-get update && apt install -y git

RUN git clone https://github.com/adalca/pystrum &&\
    git clone https://github.com/adalca/neurite &&\
    cd pystrum &&\
    pip install . &&\
    cd ../neurite &&\
    pip install . &&\
    cd /workspace &&\
    git clone https://github.com/nathandecaux/SeqSeg.git &&\
    cd SeqSeg &&\
    pip install voxelmorph-0.1-py3-none-any.whl &&\
    pip install -r /workspace/requirements.txt
    