FROM tensorflow/tensorflow:2.15.0-gpu

# Evitar prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear estructura de directorios de outputs
RUN mkdir -p outputs/visualizations outputs/checkpoints outputs/logs mappings

# Exponer puerto Jupyter
EXPOSE 8888

# Comando por defecto: bash (el Jupyter se lanza manualmente según la guía)
CMD ["bash"]