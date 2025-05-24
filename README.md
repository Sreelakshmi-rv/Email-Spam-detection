# Email-Spam-detection
This project implements an Email Spam Detection system using a Naïve Bayes classifier. The model is trained on a dataset of emails and can classify new emails as either spam or ham (not spam) using a simple web interface built with Streamlit.

# Installation
To set up the project, follow these steps:

1. Clone the repository
   '''git clone https://github.com/Sreelakshmi-rv/Email-Spam-detection.git
   cd Email-Spam-detection'''


3. Install the required dependencies
   '''pip install -r requirements.txt'''

#Usage

1. Train the Model:
   Run the spam_ham_classify.py script to train the Naïve Bayes model on the dataset. Ensure that the dataset file (spam (1).csv) is correctly specified in the script.
   '''python spam_ham_classify.py'''

2. Run the Streamlit App:
   After training the model, run the Streamlit application to classify emails.
   '''streamlit run app.py'''

3. Classify Emails:
   Open the web interface in your browser, enter the email content in the text area, and click the "Classify" button to see if the email is spam or ham.

4. Live Application:
   You can also access the live Streamlit application here- https://email-spam-detection-grserxnqxeenspkgtcfff6.streamlit.app/

#Files

spam_ham_classify.py: Script to train the Naïve Bayes model using a dataset of emails. It processes the data, trains the model, and saves it as spam_model.pkl.

app.py: Streamlit application that provides a user interface for classifying emails. It loads the trained model and allows users to input email content for classification.

requirements.txt: Lists the required Python packages for the project.

#Dependencies

The following Python packages are required to run this project:
streamlit
numpy
pandas
scikit-learn
pyngrok
You can install all dependencies using the command:
'''pip install -r requirements.txt'''
