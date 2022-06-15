#!/usr/bin/python3
import spotipy
from flask import Flask, url_for, session, request, redirect, render_template
import time
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# App config
app = Flask(__name__)

app.secret_key = '231234dsadas'
app.config['SESSION_COOKIE_NAME'] = 'My Cookies'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    # print(auth_url)
    return redirect(auth_url)


@app.route('/authorize', methods=["GET", "POST"])
def authorize():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    loginbutton = "Logged In"
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    print(sp.current_user())
    return render_template("index.html", loginbutton=loginbutton)


@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')


@app.route('/getPlaylist', methods=["GET", "POST"])
def getPlaylist():
    session['token_info'], authorized = get_token()
    session.modified = True
    if not authorized:
        return redirect('/')
    if request.method == "POST":
        playlist_name = request.form.get("playlistName")
        playlist_link = request.form.get("playlistURL")
    print("Creating {} from {}".format(playlist_name, playlist_link))

    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))

    username = sp.current_user()['href'].split('/')[5]
    print(username)
    # split it just to get the uri
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    # get the first uri
    uri = sp.playlist_tracks(playlist_URI)["items"][0]["track"]["uri"]
    # create dataframe and input the first
    df = pd.DataFrame(sp.audio_features(uri))
    # loop through and add the rest with the artist info
    track_name = []
    artist_name = []
    artist_pop = []
    track_pop = []
    for track in sp.playlist_tracks(playlist_URI)["items"]:
        uri = track["track"]["uri"]
        spi = sp.audio_features(uri)
        df = df.append(other=pd.DataFrame(spi), ignore_index=True)
        # artist info grep
        track_name.append(track["track"]["name"])
        artist_uri = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        artist_name.append(track["track"]["artists"][0]["name"])
        artist_pop.append(artist_info["popularity"])
        track_pop.append(track["track"]["popularity"])
    # take the first row off and reset the index
    df = df.drop_duplicates(ignore_index=True)
    # add the other columns
    df['track_name'] = track_name
    df['artist_name'] = artist_name
    df['artist_pop'] = artist_pop
    df['track_pop'] = track_pop
    # remove the unnecessary data
    drop_cols = ['mode', 'time_signature', 'analysis_url', 'uri', 'track_href', 'type', 'duration_ms']
    df = df.drop(drop_cols, axis=1)
    # reorder the columns
    df = df[['track_name', 'artist_name', 'artist_pop', 'track_pop', 'id', 'danceability', 'energy', 'key', 'loudness',
             'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    # using the Spotipy rec api to get more data on songs similiar to the ones in the track (this can be skipped)
    rec_song = []
    for i in df['id'].values.tolist():
        rec_song += sp.recommendations(seed_tracks=[i], limit=5)['tracks']
    rec_song_id = []
    rec_song_name = []
    rec_artist_name = []
    rec_artist_pop = []
    rec_song_pop = []
    for i in rec_song:
        rec_song_id.append(i['id'])
        rec_song_name.append(i['name'])
        rec_artist_name.append(i["artists"][0]["name"])
        artist_uri = i["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        rec_artist_pop.append(artist_info["popularity"])
        rec_song_pop.append(i["popularity"])

    rec_song_features = []
    for i in range(0, len(rec_song_id)):
        rec_audio_features = sp.audio_features(rec_song_id[i])
        for track in rec_audio_features:
            rec_song_features.append(track)
    rec_song_playlist = pd.DataFrame(rec_song_features)
    rec_song_playlist['track_name'] = rec_song_name
    rec_song_playlist['artist_name'] = rec_artist_name
    rec_song_playlist['artist_pop'] = rec_artist_pop
    rec_song_playlist['track_pop'] = rec_song_pop
    rec_song_playlist = rec_song_playlist.drop(drop_cols, axis=1)
    rec_song_playlist = rec_song_playlist[
        ['track_name', 'artist_name', 'artist_pop', 'track_pop', 'id', 'danceability', 'energy', 'key', 'loudness',
         'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    # remove duplicates
    df.drop_duplicates(subset=['id'])
    # add the favorite column and add 1 to all of the values
    df['favorite'] = 1
    # refine the rec song df
    rec_song_refined = rec_song_playlist.drop(columns=['track_name', 'artist_name', 'artist_pop', 'track_pop'])
    # read csv data for all spotify data
    spotify_df_pre = pd.read_csv('spotify_songs.csv')
    # combine the two for the entire dataframe
    frames_total = [rec_song_refined, spotify_df_pre]
    spotify_df = pd.concat(frames_total)
    # add the favorite column with zero
    spotify_df['favorite'] = 0
    # make a new data frame with just the data values (the other fave_df id just to see
    refined_fave_df = df.drop(columns=['track_name', 'artist_name', 'artist_pop', 'track_pop'])
    # combine them
    total_frames = [spotify_df, refined_fave_df]
    combined_df = pd.concat(total_frames)
    combined_df = combined_df.dropna(how='any', axis=0)
    # finally we will create the model
    # shuffle the dataset
    shuffled_c_df = combined_df.sample(frac=1)
    # make a size for the split
    split_size = int(0.8 * len(combined_df))
    # create the splits
    training_df = shuffled_c_df[:split_size]
    testing_df = shuffled_c_df[split_size:]
    # set the X and y
    X = training_df.drop(columns=['id', 'favorite'])
    y = training_df['favorite']
    # refine the test set
    X_test = testing_df.drop(columns=['id', 'favorite'])
    y_test = testing_df['favorite']
    # resample using the Synthetic Minority Oversampling Technique to get more favorite songs since it is a ratio of
    # 500 to 8000
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X, y)
    # train to plots the confusion matrix (hyperparameters found in previous tests --> now are set)
    dt = DecisionTreeClassifier(max_depth=30).fit(X_train, y_train)
    # plot a confusion matrix
    # plot_confusion_matrix(dt, X_train, y_train)
    # created pipeline to scale te data and re
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=30))
    pipe.fit(X_train, y_train)
    Pipeline(steps=[('Scaler', StandardScaler()), ('Tree', DecisionTreeClassifier(max_depth=30))])
    pipe.score(X_test, y_test)
    # Created a threshold for user to change to get more or less songs
    predict_proba = pipe.predict_proba(combined_df.drop(['favorite', 'id'], axis=1))
    accuracy = 0.999  # threshold can be changed depending on user
    length_of_predict_proba = len(predict_proba)  # length
    preds = [1 if predict_proba[i][1] > accuracy else 0 for i in range(length_of_predict_proba)]
    combined_df['prediction'] = preds
    # create the dataframe with 0 favorite (in database not OG playlist ) and prediction of 1 (songs predicted to fit
    # the model)
    playlist_df = combined_df[(combined_df['favorite'] == 0) & (combined_df['prediction'] == 1)]
    # make them a list to pass in a future function
    playlist_track_list = playlist_df['id'].tolist()
    # create the playlist in the users spotify
    playlist = sp.user_playlist_create(user=username, name=playlist_name)
    # get the ID for the playlist to add songs
    playlist_id = playlist['id']
    # add the songs
    playlist_final = sp.playlist_add_items(playlist_id, playlist_track_list)
    # get the url for the playlist
    playlist_url = (sp.current_user_playlists()['items'][0]['external_urls']).values()
    url = "https://open.spotify.com/embed/playlist/" + str(playlist_url)[48:].removesuffix(
        "'])") + "?utm_source=generator"
    loginbutton = "Logged In"
    return render_template("index.html", url=url, loginbutton=loginbutton)


# Checks to see if token is valid and gets a new token if not
def get_token():
    token_valid = False
    token_info = session.get("token_info", {})

    # Checking if the session already has a token stored
    if not (session.get('token_info', False)):
        token_valid = False
        return token_info, token_valid

    # Checking if token has expired
    now = int(time.time())
    is_token_expired = session.get('token_info').get('expires_at') - now < 60

    # Refreshing token if it has expired
    if is_token_expired:
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(session.get('token_info').get('refresh_token'))

    token_valid = True
    return token_info, token_valid


def create_spotify_oauth():
    return SpotifyOAuth(
        client_id="f6ade21d86f6443891372a007434a75f",
        client_secret="df4a7b00a1a64cb7b4b5c4dfa538b095",
        redirect_uri=url_for('authorize', _external=True),
        scope="user-top-read playlist-modify-private playlist-modify-public user-library-read")


if __name__ == "__main__":
    app.run(debug=True)
