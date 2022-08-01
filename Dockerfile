FROM docker.io/huggingface/transformers-pytorch-gpu:3.1.0



COPY . /workspace
WORKDIR /workspace

RUN pip3 install -r requirements.txt

RUN curl https://tulap.cmpe.boun.edu.tr/staticFiles/relx_finetuned_model.pt --output relx_finetuned_model.pt

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]

# CMD tail -f /dev/null
