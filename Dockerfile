FROM python:alpine
WORKDIR /app
RUN pip install redis
ADD ggbot /app/ggbot
ADD example*.py /app/
CMD ["python", "example_flobot.py", "-c", "/config.json"]
