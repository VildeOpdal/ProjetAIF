# ProjetAIF : Movie Poster Genre Classification and Recommendation System

Group members:
* Alami Ghali
* Caputo Oriane
* Nguyen Chung 
* Opdal Vilde 
* Vazquez-Arellano Laura Karina 

This project implements a movie genre classification system, along with content-based recommendation systems using posters and text. It also includes anomaly detection to distinguish valid movie posters from other images.

##  Setup and Execution
1. Clone the Repository
2. Build and Run with Docker Compose

Run the following command to build and start the application: docker-compose up --build

3. Access the Application
* Gradio Web Interface: http://localhost:7860
* API Endpoints: http://localhost:5001

4. Application Usage
A. Genre Classification
  1. Upload a movie poster.
  2. The system will classify the genre. If the uploaded image isn't a movie poster, an anomaly message is displayed.

B. Poster-Based Recommendations
  1. Upload a movie poster.
  2. Receive the 5 most visually similar movies.

C. Plot-Based Recommendations
  1. Enter a movie description.
  2. Receive the 5 most similar movies.
