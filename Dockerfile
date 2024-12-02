# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the app files into the container
COPY . /app

# Copy the model file into the container (Ensure titanic_model.pkl exists in the same directory as Dockerfile)
COPY titanic_model.pkl /app/titanic_model.pkl

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 for FastAPI
EXPOSE 80

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
