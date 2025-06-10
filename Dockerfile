
# Use NVIDIA CUDA 12.8.0 with cuDNN development tools on Ubuntu 22.04
FROM slabstech/dwani-vllm


# Set working directory
WORKDIR /app


# Copy the rest of the application code
COPY . .

RUN pip install slowapi pydantic_settings num2words

# Create a non-root user and set ownership
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app


# Switch to non-root user
USER appuser


# Expose port for the API
EXPOSE 7860


# Run the TTS API server
CMD ["python", "/app/src/gh200/main.py", "--host", "0.0.0.0", "--port", "7864", "--config", "config_two"]