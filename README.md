# email-summarization
A module for E-mail Summarization which uses clustering of skip-thought sentence embeddings.<br>
This code in this repository compliments [this Medium article](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1).
## Instructions
- The code is written in Python 2. 
- The module uses code of the [Skip-Thoughts paper](http://arxiv.org/abs/1506.06726) which can be found [here](https://github.com/ryankiros/skip-thoughts). Do:
  ```
  git clone https://github.com/ryankiros/skip-thoughts
  ```
- The code for the skip-thoughts paper uses [Theano](http://deeplearning.net/software/theano/install.html). Make sure you have Theano installed and GPU acceleration is functional for faster execution. 
- Clone this repository and copy the file `email_summarization.py` to the root of the cloned skip-thoughts repository. Do:
  ```
  git clone https://github.com/jatana-research/email-summarization
  cp email-summarization/email_summarization.py skip-thoughts/
  ```
- Install dependencies. Do:
  ```
  pip install -r email-summarization/requirements.txt
  python -c 'import nltk; nltk.download("punkt")'
  ```
- Download the pre-trained models. The total download size will be of around 5 GB. Do:
  ```
  mkdir skip-thoughts/models
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/utable.npy
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/btable.npy
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
  wget -P ./skip-thoughts/models http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
  ``` 
- Verify the MD5 hashes of the downloaded files to ensure that the files haven't been corrupted during the download. Do:
  ```
  md5sum skip-thoughts/models/*
  ```
  The output should be:
  ```
  9a15429d694a0e035f9ee1efcb1406f3 bi_skip.npz
  c9b86840e1dedb05837735d8bf94cee2 bi_skip.npz.pkl
  022b5b15f53a84c785e3153a2c383df6 btable.npy
  26d8a3e6458500013723b380a4b4b55e dictionary.txt
  8eb7c6948001740c3111d71a2fa446c1 uni_skip.npz
  e1a0ead377877ff3ea5388bb11cfe8d7 uni_skip.npz.pkl
  5871cc62fc01b79788c79c219b175617 utable.npy
  ```
- Change `Lines:23-24` in the file `skip-thoughts/skipthoughts.py` to provide the correct paths to the downloaded models.
  ```
  path_to_models = 'models/'
  path_to_tables = 'models/'
  ```
  
## Running the module
- Find any English emails dataset online or create a small one on your own.
- The module expects a list of emails as input and returns a list of summaries.
- Open the Python interpreter in the `skip-thoughts/` folder and do:
  ```
  >>> from email_summarization import summarize
  >>> summaries = summarize(emails) # emails is a Python list containing English emails.
  ```
