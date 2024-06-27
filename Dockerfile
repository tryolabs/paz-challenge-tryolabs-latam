FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

# Install the required Python packages
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

# Copy all files from the current directory to the working directory in the container
COPY . .

# Expose port 8080 of the container to external network
EXPOSE 8080

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
