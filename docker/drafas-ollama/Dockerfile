FROM ollama/ollama:0.3.11
# need to run ollama process first before pull
RUN ollama serve & sleep 2 && \
    ollama pull llama3.2:1b-instruct-q4_K_M
RUN apt update && apt install -y curl
ENV OLLAMA_NUM_PARALLEL=1

