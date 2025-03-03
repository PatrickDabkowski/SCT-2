FROM pytorch/pytorch

WORKDIR /home

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    numpy SimpleITK lungmask

COPY . /home

ENTRYPOINT ["python3", "main.py"]
CMD ["main.py"]