# Container for the FR3 / D-MIMO power-model GUI.
#
# Build context is the code/ directory, so the image bundles the GUI together
# with the two model packages it imports (fr3_power and D_MIMO_rate). Only the
# light dependencies are installed: the GUI reads cached rates and never runs
# the Sionna/TensorFlow rate model, so those are deliberately left out.
#
# Build and run (from the code/ directory):
#   docker build -t fr3-dmimo-power-gui .
#   docker run --rm -p 8501:8501 fr3-dmimo-power-gui
# then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached across code changes.
COPY power_gui/requirements.txt /app/power_gui/requirements.txt
RUN pip install --no-cache-dir -r power_gui/requirements.txt

# Copy the GUI and the model packages it imports (see power_gui/model_access.py).
COPY FR3_power_model/ /app/FR3_power_model/
COPY D_MIMO_rate/ /app/D_MIMO_rate/
COPY power_gui/ /app/power_gui/

# Headless server, non-interactive Matplotlib backend, no usage telemetry.
ENV MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "power_gui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
