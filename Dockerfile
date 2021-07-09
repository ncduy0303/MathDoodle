FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]