
import re
import nltk
from nltk.corpus import stopwords
import string


# Download the stopwords corpus from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Define the preprocessing function
def preprocess_message(message):
    
    Message = message.lower()
    Message = re.sub('\[.*?\]', '', Message)
    Message = re.sub('https?://\S+|www\.\S+', '', Message)
    Message = re.sub('<.*?>+', '', message)
    Message = re.sub('[%s]' % re.escape(string.punctuation), '', Message)
    Message = re.sub('\n', '', Message)
    Message = re.sub('\w*\d\w*', '', Message)
    
    stop_words = set(stopwords.words('english'))
    message = ' '.join([word for word in message.split() if word not in stop_words])
    return message