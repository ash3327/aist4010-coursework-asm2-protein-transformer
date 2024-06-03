## Procedures

1. Download the dataset either by direct downloading, or through kaggle by the following:

    a. Generate the Kaggle api key from kaggle / accounts / generate api key.
    
    b. Put the kaggle.json generated into the folder specified by the error message generated when you execute ```kaggle competitions download -c aist4010-spring2024-a2```.

    c. Execute ```kaggle competitions download -c aist4010-spring2024-a2``` in command prompt and unzip the file in any manner. Make sure you unzipped it with root folder containing the directory ```aist4010-spring2024-a2/data```.

2. Execute ```pip3 install -r requirements.txt``` in command prompt, and also install PyTorch that matches your needs. You may want to install PyTorch versions compatible with the CUDA and GPU you're using.