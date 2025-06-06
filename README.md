# COMP702-Project

A Coin Classification research project done on old and new post-democracy South African coins. The coin_dataset was augmented via the augment.py script in legacy_code folder. The CoinClassifier Folder consists of the code done in AndroidStudio to create a deployable apk file.

Running Instructions:

1. Android device
-Navigate to the Release section and download the apk file. 
-Open on your android device to install the application. 
-After installation simply open the app to begin classifying either by taking a photo or loading an image.

2. Windows Device
-Navigate to the Release section and download the windows zip file. 
-Extract to a directory of your choice and then enter the folder. Python must also be installed.
-Double click the bat file to execute the program. 
-Load an image and click the classify button to classify it.

Coin dataset:
The dataset was compiled using the South African subsection on the OnlineCoinClub (https://onlinecoin.club/Coins/Country/South_Africa/) website as the main repository. The website documents each coin denomination throughout all the relevant years (1994 - 2022). The newer coins, released in 2023 were sourced from other loyalty-free images in order to fully represent all the South African coin variations.
After downloading the front and back images of each coin, the dataset was cleaned to remove any class imbalances in some coins, resulting in 295 coin images. Following augmentation on the the images, the final dataset consists of 3245 coin images. 
