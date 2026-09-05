FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY ui ./ui
COPY submission ./submission
COPY configs ./configs
# the original deck powers the "rebuild from the deck" reproducibility check
COPY aios-track2/materials/41_Model_Z_final_OPM.zip ./aios-track2/materials/41_Model_Z_final_OPM.zip
RUN pip install --no-cache-dir fastapi 'uvicorn[standard]' numpy scipy pydantic pandas pyyaml \
    && pip install --no-cache-dir --no-deps .
EXPOSE 8765
ENV PYTHONPATH=src
CMD ["python", "-m", "aios_track2.cli", "ui", "--host", "0.0.0.0", "--port", "8765", "--submission", "submission"]
