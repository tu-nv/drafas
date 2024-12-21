FROM ghcr.io/coqui-ai/tts

# prefetch the model
RUN timeout --preserve-status 30s python3 TTS/server/server.py --model_name tts_models/en/vctk/vits || true

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

