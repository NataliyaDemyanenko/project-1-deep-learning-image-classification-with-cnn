# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the saved_model.pb file into the container at /app
COPY saved_model.pb /app/saved_model.pb

# Define the command to run your TensorFlow model
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=saved_model", "--model_base_path=/app"]


