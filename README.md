# Curating New Spotify Playlists through Machine Learning Models
In this project I used the Spotipy api to parse and analyze user's playlists and curate a new playlist using a Decision Tree Classifier model. Moreover, to make user accessiblity simpler, I created a website using HTML/CSS, Javascript, and Flask (pictures below)
## Prerequisites

Before you run this program, follow these steps:
1. If you don't have a python enviroment, make sure to install it first (in the terminal)
```
python3 -m venv local_python_environment
```
2. Run the requirments for the software (in the terminal):
```
pip install -r requirements.txt
```

## Running the Program

To run the program simply do these steps
1. Set the flask app to the main python file 
```
set FLASK_APP=app.py
```
2. Run the flask app 
```
flask run
```
This image should show up:

<img width="503" alt="image" src="https://user-images.githubusercontent.com/56817389/173717210-e996beee-a525-4e84-9dd8-4f109055dd27.png">

## Website 
Here is how the website looks like:

![screen-recording](https://user-images.githubusercontent.com/56817389/173719171-e9a68c75-ef67-417c-8345-cc3ae03b63ea.gif)


Make sure to login to Spotify before making the playlist (should take you to this browser):

<img width="537" alt="image" src="https://user-images.githubusercontent.com/56817389/173717551-3ba167bd-3819-486f-a688-f36d158b9458.png">

Once you logged in, you can type in a name for your new playlist and the link to the playlist the ML model should get reccomendations off of and press submit

The playlist will be automatically saved in your library as well as showing up on the website like this:

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/56817389/173717929-1faf1710-81e3-4466-aa07-d90b142d444c.png">

You can also log out of your account once you are finished, otherwise your login information will be saved if you ever want to make another playlist 

## Source Code

If you want a deeper look into the model and the accuracy of the model, run the juypter notebook in the Juypter file which will give you several metrics on the models performance

## Contact

If you want to contact me for any help you can reach me at nagaredd@usc.edu.

