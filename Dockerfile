# Use an official Python runtime as the parent image
FROM python:3.8
# Set the working directory in the container to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Make port 7860 available to the world outside this container
# Gradio by default runs on port 7860
EXPOSE 7860
# Run the Gradio app when the container launches
CMD ["python", "gradio_webapp.py"]