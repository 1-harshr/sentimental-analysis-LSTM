# Sentimental Analysis using BiLSTM
Classify as `pos`, `neg` or `neutral`

The code provided is a Python script that performs various tasks using machine learning and natural language processing libraries. The script is intended for use on a GitHub repository.

## Dependencies

The code requires the following dependencies:

- `random`
- `copy`
- `time`
- `pandas`
- `numpy`
- `gc`
- `re`
- `torch`
- `keras.preprocessing.text`
- `tqdm`
- `collections.Counter`
- `nltk`
- `torch.nn`
- `torch.optim`
- `torch.nn.functional`
- `torch.utils.data`
- `torch.nn.utils.rnn`
- `torch.autograd.Variable`
- `sklearn.metrics.f1_score`
- `os`
- `keras.utils.pad_sequences`
- `sklearn.model_selection.StratifiedKFold`
- `sklearn.metrics.f1_score`
- `torch.optim.optimizer.Optimizer`
- `sklearn.preprocessing.StandardScaler`
- `multiprocessing.Pool`
- `functools.partial`
- `sklearn.decomposition.PCA`
- `matplotlib.pyplot`

## Data Loading

The code reads a CSV file named "train.csv" using `pandas.read_csv`. The loaded data is stored in a variable named `data`. It is important to note that the code assumes the existence of this file in the specified location.

## Data Preprocessing

The code performs several preprocessing steps on the loaded data:

1. Lowercasing: The text in the "sentence" column of the data is converted to lowercase using the `lower()` method.
2. Cleaning Text: The `clean_text` function is applied to remove special characters from the text.
3. Cleaning Numbers: The `clean_numbers` function is applied to replace numbers with special tokens.
4. Cleaning Contractions: The `replace_contractions` function is used to expand contractions in the text.
5. Removal of Non-Word Characters: The text is further cleaned by removing non-word characters using regular expressions.

## Tokenization and Padding

The code performs tokenization and padding of the text data using the `Tokenizer` class from `keras.preprocessing.text` and the `pad_sequences` function from `keras.utils`. The `Tokenizer` is fit on the training data and then used to convert the text data into sequences of tokens. The sequences are padded to a fixed length specified by the `maxlen` variable.

## Data Splitting

The code splits the data into training and validation sets using the `train_test_split` function from `sklearn.model_selection`. The training data is stored in the `train_data` variable, and the validation data is stored in the `val_data` variable.

## Label Encoding

The code performs label encoding on the target variables using the `LabelEncoder` class from `sklearn.preprocessing`. The encoded labels are stored in the `train_y` and `test_y` variables.

## LSTM Model Definition

The code defines an LSTM (Long Short-Term Memory) model using the `nn.Module` class from `torch.nn`. The model architecture consists of an embedding layer, an LSTM layer, dropout layers, and fully connected layers. The model is designed to classify text into three classes.

## Training

The code trains the LSTM model using the training data. It uses the Adam optimizer and the BCEWithLogitsLoss loss function from `torch.nn` to optimize the model parameters. The training is performed over multiple epochs, with each epoch consisting of forward pass, loss computation, backward pass, and parameter update steps. The model's performance on the validation data is evaluated after each epoch.

## Plotting Loss

The code plots the training and validation loss curves using `matplotlib.pyplot`.

## Additional Preprocessing

The code performs additional preprocessing steps on the text data:

1. Stopword Removal: The code removes stopwords from the text using the `stopwords.words('english')` function from `nltk`. Stopwords are common words that do not carry much meaning and are often removed to reduce noise in the data.
2. Lemmatization: The code uses the WordNet lemmatizer from `nltk` to convert words to their base or dictionary form. This step helps standardize different forms of the same word, such as converting "running" to "run" or "mice" to "mouse".

## Model Evaluation

The code evaluates the trained LSTM model on the test data using the F1 score metric from `sklearn.metrics`. The F1 score provides a measure of the model's performance by considering both precision and recall. The evaluation results are printed, including the F1 score for each class and the overall weighted F1 score.

## Conclusion

The code provided showcases a complete pipeline for text classification using an LSTM model.
