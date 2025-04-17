FROM python:3.10-slim-bullseye

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    make \
    git \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install latest SQLite manually (>= 3.35)
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz && \
    tar -xzf sqlite-autoconf-3420000.tar.gz && \
    cd sqlite-autoconf-3420000 && \
    ./configure --prefix=/usr/local && make && make install && \
    cd .. && rm -rf sqlite-autoconf-3420000*

# Set environment so Python uses the new SQLite
ENV LD_LIBRARY_PATH="/usr/local/lib"
ENV PATH="/usr/local/bin:$PATH"

# Confirm sqlite version
RUN sqlite3 --version  # Should print 3.42.0

# Set working dir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "streamlit_app.py"]
