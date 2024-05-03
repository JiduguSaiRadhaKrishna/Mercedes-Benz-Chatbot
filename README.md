# Mercedes-Benz-Chatbot
QA Chatbot
---
This project implements a simple chatbot using Flask for the web interface, TensorFlow/Keras for the neural network model, and NLTK for text preprocessing. The chatbot retrieves data from a Wikipedia page, preprocesses it, trains a TF-IDF vectorizer and a neural network model, and then responds to user queries based on the trained model.

## Setup Instructions
Prerequisites
-Python 3.x installed on your system
-pip package manager (usually comes with Python installation)
-Virtual environment (optional but recommended)
### Prerequisites
- Python 3.x installed on your system
- `pip` package manager (usually comes with Python installation)
- Virtual environment (optional but recommended)
### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/chatbot-flask-tensorflow.git
   ```
2. Navigate to the project directory:
  ```bash
   cd chatbot-flask-tensorflow
   ```
3. Create a virtual environment (optional):
 ```bash
   python -m venv venv
  ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
      ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Running the Application
1. Fetch Wikipedia data and train the chatbot model:

   ```bash
   python main.py
   ```
2. Once the model is trained and ready, run the Flask web application:
   ```bash
   python main.py
   ```
3. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the chatbot interface.

### Usage
- Enter your question or query into the input field on the web page.
- Click the "Submit" button to send your query to the chatbot.
- The chatbot will process your query and display the response on the web page.
## Project Structure
- `main.py`: Main Flask application script that handles web routes, model training, and inference.
- `templates/`: HTML templates for rendering web pages.
  - `index.html`: Main HTML file containing the chatbot interface.
- `README.md`: Documentation file explaining the project and setup instructions.
- `requirements.txt`: List of Python packages required for the project with specific versions.
## Additional Notes
- The chatbot currently uses a basic neural network with TF-IDF vectorization for question-response matching.
- For improved performance, consider experimenting with more sophisticated models, larger datasets, and advanced preprocessing techniques.
---
