#!/usr/bin/env python3
# Your Own Twitter Filter Bubble
# Learn your interests from your Twitter favorites, tweets and retweets and
# filter/reorder the timeline accordingly.

import sys
import argparse
from datetime import datetime
import sqlite3
import tweepy
import logging
import itertools
import math
import pickle
import re
import json
from keras import utils, preprocessing, models, layers, callbacks
from keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger()

tweepy_api_instance = None

### Tools ###
# Following class is copied from https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def datetimearg(dtstr):
    """Used by argument parser for parsing of dates and times."""
    try:
        return datetime.strptime(dtstr, "%Y-%m-%d")     # first try to parse date only
    except ValueError:
        return datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")     # user may have given a time, try this

def get_twitter_api():
    """Get an authenticated instance of a Tweepy API interface"""
    global tweepy_api_instance

    if tweepy_api_instance is None:
        auth = tweepy.OAuthHandler(args.consumer_key, args.consumer_secret)
        auth.set_access_token(args.access_token, args.access_token_secret)
        tweepy_api_instance = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return tweepy_api_instance

def tweet_sql_values_generator(tweets):
    """Generate values tuple for SQL INSERT statement from list of tweet objects."""
    twitter = get_twitter_api()
    user = twitter.me()
    i = 0
    for tweet in tweets:
        i += 1
        yield (tweet.id, tweet.created_at, tweet.user.id, tweet.text, tweet.favorited, tweet.retweeted, tweet.user.id == user.id and not tweet.retweeted)
    logger.debug("Processed %d tweets", i)

def log_histogram(a):
    hist = np.histogram(a)
    for c, i in zip(hist[0], hist[1]):
        if c > 0:
            logger.debug("Count %.2f..%.2f = %d" % (i, i + 0.1, c))

### Tweet acquisition ###
def get_timeline():
    """Get timeline. Fetch from latest stored tweet."""
    dbc = db.cursor()
    dbc.execute("SELECT MAX(id) FROM tweets")
    max_id = dbc.fetchone()[0]
    if max_id is not None:
        logger.debug("Fetching timeline tweets with id > %d", max_id)
    else:
        logger.debug("Fetching as much timeline tweets as possible")
    twitter = get_twitter_api()
    return tweepy.Cursor(twitter.home_timeline, count=args.fetchcount, since_id=max_id).items(args.max_timeline_tweets)

def get_favorites():
    twitter = get_twitter_api()
    """Get user favorites"""
    return tweepy.Cursor(twitter.favorites).items(args.max_favorites)

def get_own():
    twitter = get_twitter_api()
    """Get user tweets"""
    return tweepy.Cursor(twitter.user_timeline, count=args.fetchcount).items(args.max_own_tweets)

### Deep Learning: Models and input generation ###
# Input sequence generator classes
class TweetSequenceBase(utils.Sequence):
    """
    Generate training batches from database query result. This base class initializes
    a Tokenizer fitted to text from database API cursor and calls the method create_input_sequences
    with all input texts and should return the input the target model expects.

    The class further calculates the label according to the given scoring parameters.
    """
    def __init__(self, tweets, words, exclude_urls, exclude_mentions, exclude_answers, batch_size, favorite_score, retweet_score, owntweet_score, k):
        tweet_texts = [ tweet['text'] for tweet in tweets ]
        tweet_ids = [ tweet['id'] for tweet in tweets ]
        self.favorite_score = favorite_score
        self.retweet_score = retweet_score
        self.owntweet_score = owntweet_score
        self.first_id = min(tweet_ids)
        self.last_id = max(tweet_ids)
        if exclude_urls:
            reURL = re.compile("https?://\S+")
            for i, tweet in enumerate(tweet_texts):
                tweet_texts[i] = reURL.sub("", tweet)
        if exclude_mentions:
            reMention = re.compile("@[a-z0-9_]+", re.IGNORECASE)
            for i, tweet in enumerate(tweet_texts):
                tweet_texts[i] = reMention.sub("", tweet)
        if exclude_answers:
            before = len(tweet_texts)
            tweet_texts = list(filter(lambda tweet: not tweet.startswith("@"), tweet_texts))
            after = len(tweet_texts)
            if after < before:
                logger.debug("Reduced input set from %d to %d tweets by removing answers" % (before, after))
            else:
                logger.warn("Answer filtering hasn't reduced input set.")
                
        tokenizer = Tokenizer(num_words=words)
        tokenizer.fit_on_texts(tweet_texts)
        self.tokenizer = tokenizer
        self.update_input(tweet_texts)
        self.update_output(tweets)
        logger.debug("Shapes: x=%s y=%s", self.x.shape, self.y.shape)
        logger.debug("Input label histogram:")
        log_histogram(self.y)
        self.batch_size = batch_size
        self.k = k
        if k is not None:
            self.k_size = self.x.shape[0] // k
            self.shuffle()

    def shuffle(self):
        p = np.random.permutation(len(self.x))
        xs = np.zeros_like(self.x)
        ys = np.zeros_like(self.y)
        for i, j in enumerate(p):
            xs[i] = self.x[j]
            ys[i] = self.y[j]
        self.x = xs
        self.y = ys

    def save(self, f):
        pickle.dump(self, f)

    def create_input_sequences(self, tweets):
        raise NotImplementedError("Input sequence generation is not implemented in TweetSequenceBase class")

    def update_input(self, tweets):
        """Update input sequences from array of tweet strings"""
        self.x = self.create_input_sequences(tweets)
        self.y = None

    def update_output(self, tweets):
        """
        Update output score vector for a list of dicts with following keys:

        * favorite: 1 if tweet was favorited
        * retweet: 1 if tweet was retweeted by used account
        * own: 1 if tweet was posted by used account
        """
        self.y = np.array(
                [
                    max([
                        tweet['favorite'] * self.favorite_score,
                        tweet['retweet'] * self.retweet_score,
                        tweet['own'] * self.owntweet_score
                        ])
                    for tweet in tweets
                ]
                )

    def update(self, tweets):
        """
        Update input generator from list of dicts with the following keys:

        * text: Tweet text
        * Remaining as in update_output
        """
        self.update_input([ tweet['text'] for tweet in tweets ])
        self.update_output(tweets)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, i):
        return (
                np.array(self.x[i * self.batch_size:(i+1) * self.batch_size]),
                np.array(self.y[i * self.batch_size:(i+1) * self.batch_size]),
                )

    def get(self, a, val=None, i=None):
        if val:                 # return validation set in k-fold-validation
            return a[i * self.k_size:(i + 1) * self.k_size]
        elif val == False:      # return partial train set in k-fold-validation
            return np.concatenate([
                a[:i * self.k_size],
                a[(i + 1) * self.k_size:]
                ])
        else:                   # return complete data
            return a

    def get_x(self, val=None, i=None):
        return self.get(self.x, val, i)

    def get_y(self, val=None, i=None):
        return self.get(self.y, val, i)

    def max_len(self):
        return self.x.shape[1]

    def dump_tokenizer(self):
        return pickle.dumps(self.tokenizer)

    def dump_dictionary(self):
        return [ k for k, v in sorted(self.tokenizer.word_index.items(), key=lambda i: i[1])]

class OneHotTweetSequence(TweetSequenceBase):
    """Create an one-hot encoded sequence per Tweet."""
    name = "One hot input generator"
    def create_input_sequences(self, tweets):
        return self.tokenizer.texts_to_matrix(tweets, mode='binary')

class PaddedSequencesTweetSequence(TweetSequenceBase):
    """Create padded word-index sequences for each Tweet"""
    name = "Padded word-index sequence generator"
    def create_input_sequences(self, tweets):
        try:
            return preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences(tweets), self.max_len())
        except AttributeError:
            return preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences(tweets))

# Model generator functions
def generate_dense_model(vocabsize, units, layercnt, dropout, activation, final_activation, optimizer, loss, metrics):
    model = models.Sequential()
    for i in range(layercnt):
        if i == 0:
            model.add(layers.Dense(units, activation=activation, input_dim=vocabsize))
        else:
            model.add(layers.Dense(units, activation=activation))
        if dropout is not None:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation=final_activation))

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            )
    return model

def generate_embedding_dense_model(maxwords, vocabsize, units, layercnt, embedding_units, dropout, activation, final_activation, optimizer, loss, metrics):
    model = models.Sequential()
    model.add(layers.Embedding(vocabsize, embedding_units, input_length=maxwords))
    model.add(layers.Flatten())
    for i in range(layercnt):
        if dropout is not None:
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units, activation=activation))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation=final_activation))

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            )
    return model

def generate_gru_model(maxwords, vocabsize, units, layercnt, recurrent_layercnt, embedding_units, recurrent_units, dropout, recurrent_dropout, recurrent_activation, activation, final_activation, optimizer, loss, metrics):
    model = models.Sequential()
    model.add(layers.Embedding(vocabsize, embedding_units, input_length=maxwords))
    for i in range(recurrent_layercnt):
        model.add(layers.GRU(recurrent_units, activation=recurrent_activation, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=(i < recurrent_layercnt - 1)))
    for i in range(layercnt):
        if dropout is not None:
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units, activation=activation))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation=final_activation))

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            )
    return model

def generate_lstm_model(maxwords, vocabsize, units, layercnt, recurrent_layercnt, embedding_units, recurrent_units, dropout, recurrent_dropout, recurrent_activation, activation, final_activation, optimizer, loss, metrics):
    model = models.Sequential()
    model.add(layers.Embedding(vocabsize, embedding_units, input_length=maxwords))
    for i in range(recurrent_layercnt):
        model.add(layers.LSTM(recurrent_units, activation=recurrent_activation, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=(i < recurrent_layercnt - 1)))
    for i in range(layercnt):
        if dropout is not None:
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units, activation=activation))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation=final_activation))

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            )
    return model

def generate_convnet_model(maxwords, vocabsize, units, layercnt, conv_layercnt, embedding_units, conv_filters, conv_window, pool_size, flatten_after_conv, dropout, conv_activation, activation, final_activation, optimizer, loss, metrics):
    model = models.Sequential()
    model.add(layers.Embedding(vocabsize, embedding_units, input_length=maxwords))
    for i in range(conv_layercnt):
        model.add(layers.Conv1D(conv_filters, conv_window, activation=conv_activation))
        if i < conv_layercnt - 1:   # Max pooling until last layer
            model.add(layers.MaxPool1D(pool_size))
        else:                       # Layer after last conv layer prepares output for dense layers
            if flatten_after_conv:
                model.add(layers.Flatten())
            else:
                model.add(layers.GlobalMaxPool1D())
    for i in range(layercnt):
        if dropout is not None:
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units, activation=activation))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation=final_activation))

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
            )
    return model

# Dictionary of supported models with model name (corresponding to argument parser parameter name) to tuple
# mappings. The tuples are defined as follows:
# 1. input sequence generation class
# 2. model generation function
# 3. tuple of command line argument names. Tuple position maps to positional parameters of model generation function
MODELARG_MAXLEN = 1     # Maximum sequence length
model_configs = { 
        "dense": (
            OneHotTweetSequence,
            generate_dense_model,
            ("words", "units", "layers", "dropout", "dense_activation", "final_activation", "optimizer", "loss", "metric")
            ),
        "embedding_dense": (
            PaddedSequencesTweetSequence,
            generate_embedding_dense_model,
            (MODELARG_MAXLEN, "words", "units", "layers", "embedding_units", "dropout", "dense_activation", "final_activation", "optimizer", "loss", "metric")
            ),
        "gru": (
            PaddedSequencesTweetSequence,
            generate_gru_model,
            (MODELARG_MAXLEN, "words", "units", "layers", "recurrent_layers", "embedding_units", "recurrent_units", "dropout", "recurrent_dropout", "recurrent_activation", "dense_activation", "final_activation", "optimizer", "loss", "metric")
            ),
        "lstm": (
            PaddedSequencesTweetSequence,
            generate_lstm_model,
            (MODELARG_MAXLEN, "words", "units", "layers", "recurrent_layers", "embedding_units", "recurrent_units", "dropout", "recurrent_dropout", "recurrent_activation", "dense_activation", "final_activation", "optimizer", "loss", "metric")
            ),
        "convnet": (
            PaddedSequencesTweetSequence,
            generate_convnet_model,
            (MODELARG_MAXLEN, "words", "units", "layers", "conv_layers", "embedding_units", "conv_filters", "conv_window", "pool_size", "flatten_after_conv", "dropout", "conv_activation", "dense_activation", "final_activation", "optimizer", "loss", "metric")
            ),
        }

### Commands ###
def cmd_gettweets(args):
    # Fetch
    timeline = get_timeline()
    favorites = []
    own = []
    if args.favorites or args.all:
        favorites = get_favorites()
    if args.own or args.all:
        own = get_own()
    tweets = itertools.chain(timeline, favorites, own)

    # Store
    dbc = db.cursor()
    dbc.executemany("INSERT OR REPLACE INTO tweets(id, time, user, text, favorite, retweet, own) VALUES (?, ?, ?, ?, ?, ?, ?)", tweet_sql_values_generator(tweets))
    db.commit()

def cmd_train(args):
    # Get training data and initialize input generator
    dbc = db.cursor()
    if args.hold_out is None:
        dbc.execute("SELECT id, text, favorite, retweet, own FROM tweets")
    else:
        dbc.execute("SELECT id, time, text, favorite, retweet, own FROM tweets WHERE time < ?", (args.hold_out, ))
    tweets = dbc.fetchall()
    dbc.close()
    logger.info("Training data contains %d tweets", len(tweets))

    # Select model_config chosen by model selection argument
    model_config = model_configs[next(filter(lambda k: getattr(args, k), model_configs.keys()))]
    # Extract input and model generator, create parameter list and generate model
    input_generator = model_config[0](tweets, args.words, args.exclude_urls, args.exclude_mentions, args.exclude_answers, args.batch_size, args.score_favorite, args.score_retweet, args.score_owntweet, args.k_fold)
    logger.debug("Input generator: %s", input_generator.name)
    if args.print_dictionary:
        for i, w in enumerate(input_generator.dump_dictionary()):
            print("%10d | %s" % (i, w))
    model_generator = model_config[1]
    logger.debug("Model generator: %s", model_generator.__name__)
    model_args = []
    for arg_name in model_config[2]:
        if isinstance(arg_name, str):
            arg_value = getattr(args, arg_name)
        elif arg_name == MODELARG_MAXLEN:
            arg_value = input_generator.max_len()
        logger.debug("Adding argument %s with value: %s", arg_name, arg_value)
        model_args.append(arg_value)
    logger.debug("Model initiated with following parameters: %s", model_args)
    model = model_generator(*model_args)
    model.summary()

    # Build callback list according to configuration
    callback_list = []
    modelcheckpoint_args = dict()
    if args.early_stopping:
        callback_list.append(
                callbacks.EarlyStopping(
                    monitor=args.early_stopping_metric,
                    patience=args.patience,
                    mode='auto'
                    )
                )
        modelcheckpoint_args.update({
                "monitor": args.early_stopping_metric,
                "save_best_only": True
                })
    if args.output is not None:
        # save best model from training
        modelcheckpoint_args.update({ "filepath": args.output + ".h5" })
        logger.debug("ModelCheckpoint arguments: %s", modelcheckpoint_args)
        callback_list.append(
                callbacks.ModelCheckpoint(**modelcheckpoint_args)
                )

        # save input generator with tokenizer
        f = open(args.output + ".tok", "wb")
        input_generator.save(f)
        f.close()

        # save state
        f = open(args.output + ".state", "w")
        json.dump({
            "first_tweet": input_generator.first_id,
            "last_tweet": input_generator.last_id,
            }, f)
        f.close()

    # Train model
    if args.validation_split is not None:       # validation with split factor
        x = input_generator.get_x()
        y = input_generator.get_y()
        model.fit(
                x, y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callback_list,
                validation_split=args.validation_split,
                verbose=1
                )
    elif args.k_fold is not None:               # k-fold validation
        hists = dict()
        k = args.k_fold
        for i in range(k):
            logger.info("Validating fold %d/%d", i + 1, k)
            x = input_generator.get_x(False, i)
            y = input_generator.get_y(False, i)
            logger.debug("- Training label histogram:")
            log_histogram(y)

            x_val = input_generator.get_x(True, i)
            y_val = input_generator.get_y(True, i)
            logger.debug("- Validation label histogram:")
            log_histogram(y_val)

            hist = model.fit(
                    x, y,
                    validation_data=(x_val, y_val),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=1
                    )
            for metric, values in hist.history.items():
                hists.setdefault(metric, []).append(values)
            model = model_generator(*model_args)
        means = { metric: np.mean(values, axis=0) for metric, values in hists.items() }
        i = 1
        for metric, values in means.items():
            print("Metric %s: min=%.3f max=%.3f" % (metric, min(values), max(values)))
            plt.subplot(2, 2, i)
            i += 1
            plt.title(metric)
            plt.plot(range(1, len(values) + 1), values)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
        if args.output_graph:
            plt.savefig(args.output_graph)
        else:
            plt.show()
    else:                                       # training with full data set
        x = input_generator.get_x()
        y = input_generator.get_y()
        model.fit(
                x, y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callback_list,
                verbose=1
                )

def load_data(prefix, start_time=None, print_dictionary=False):
    # load state
    f = open(prefix + ".state", "r")
    state = json.load(f)
    f.close()

    # Restore input generator
    f = open(prefix + ".tok", "rb")
    input_generator = pickle.load(f)
    f.close()
    if print_dictionary:
        for i, w in enumerate(input_generator.dump_dictionary()):
            print("%10d | %s" % (i, w))

    # load model
    model = models.load_model(prefix + ".h5")
    model.summary()

    # Get training data
    dbc = db.cursor()
    if args.start_time is None:
        dbc.execute("SELECT id, time, text, favorite, retweet, own FROM tweets WHERE id > ?", (state['last_tweet'], ))
    else:
        dbc.execute("SELECT id, time, text, favorite, retweet, own FROM tweets WHERE time >= ?", (args.start_time, ))
    tweets_dbc = dbc.fetchall()
    dbc.close()

    return input_generator, model, tweets_dbc

def cmd_predict(args):
    input_generator, model, tweets_dbc = load_data(args.prefix[0], args.start_time, args.print_dictionary)
    logger.info("Predicting most interesting from %d tweets", len(tweets_dbc))
    tweets = [ tweet['text'] for tweet in tweets_dbc ]
    times = [ tweet['time'] for tweet in tweets_dbc ]

    # Initiate input generator and predict scores
    input_generator.update_input(tweets)
    y = model.predict(input_generator.get_x())

    for tweet, time, score in zip(tweets, times, y):
        if score >= args.min_score:
            print("%1.2f | %19s | %s" % (score, time, tweet))

def cmd_evaluate(args):
    input_generator, model, tweets_dbc = load_data(args.prefix[0], args.start_time, args.print_dictionary)
    logger.info("Evaluating model performance against %d tweets", len(tweets_dbc))

    # Initiate input generator and predict scores
    input_generator.update(tweets_dbc)
    res = model.evaluate(input_generator.get_x(), input_generator.get_y())

    print("%20s | %s" % ("Metric", "Value"))
    for metric, value in zip(model.metrics_names, res):
        print("%20s | %.3f" % (metric, value))

### Main program ###
argparser = argparse.ArgumentParser(description="Your Own Twitter Filter Bubble")
argparser.add_argument("--database", "-d", default='tweets.db', help="SQLite database used internally to store Tweets. Will be created if not already existing. Default: %(default)s")
argparser.add_argument('--debug', '-D', action='store_true', help="Enable debug output")
argparser.add_argument('--print-dictionary', '-pd', action='store_true', help="Print dictionary generated by tokenizer from tweets")
subargparsers = argparser.add_subparsers(dest='command', help="Commands")

tweetargparser = subargparsers.add_parser('gettweets', help="Get tweets. By default, get timeline beginning with Tweet ID from last run.")
tweetargparser.add_argument('--config', '-c', type=open, action=LoadFromFile, help="Tweet aquisition configuration file")
tweetargparser.add_argument('--consumer-key', help="Twitter API Consumer Key (not recommended - sensible data may be stored in shell history!)")
tweetargparser.add_argument('--consumer-secret', help="Twitter API Consumer Secret (not recommended - sensible data may be stored in shell history!)")
tweetargparser.add_argument('--access-token', help="Twitter API Access Token (not recommended - sensible data may be stored in shell history!)")
tweetargparser.add_argument('--access-token-secret', help="Twitter API Access Token Secret (not recommended - sensible data may be stored in shell history!)")
tweetargparser.add_argument('--fetchcount', '-n', type=int, default=200, help="Number of tweets fetched per API request (Default: %(default)d)")
tweetargparser.add_argument('--max-timeline-tweets', '-T', type=int, default=25000, help="Maximum number of fetched tweets (default: %(default)d)")
tweetargparser.add_argument('--max-favorites', '-F', type=int, default=5000, help="Maximum number of fetched favorites (default: %(default)d)")
tweetargparser.add_argument('--max-own-tweets', '-O', type=int, default=5000, help="Maximum number of fetched tweets of authenticated user (default: %(default)d)")
tweetargparser.add_argument('--favorites', '-f', action='store_true', help="Get user accounts favorites instead of timeline. Does not store tweets which are already stored in database.")
tweetargparser.add_argument('--own', '-o', action='store_true', help="Get user accounts tweets instead of timeline. Does not store tweets which are already stored in database.")
tweetargparser.add_argument('--all', '-a', action='store_true', help="Shortcut for -fro - get favorites and own (re)tweets instead of timeline and store missing tweets in database.")

trainargparser = subargparsers.add_parser('train', help="Train neuronal networks with aquired tweet data.")
trainargparser.add_argument('--config', '-c', type=open, action=LoadFromFile, help="Training configuration file")
trainargparser.add_argument('--output', '-o', help="File name prefix for stored model (.h5), configuration (.tok) and state (.state)")
trainargparser.add_argument('--output-graph', '-og', help="File for graph output. Format is deduced from extension.")

inputgroup = trainargparser.add_argument_group(title="Input Configuration", description="Modify input data")
inputgroup.add_argument('--exclude-urls', '-xu', action='store_true', help="Exclude URLs from dictionary")
inputgroup.add_argument('--exclude-mentions', '-xm', action='store_true', help="Exclude mentions (@handle) from dictionary")
inputgroup.add_argument('--exclude-answers', '-xa', action='store_true', help="Exclude answers (tweets beginning with @) from training set")
inputgroup.add_argument('--hold-out', '-ho', type=datetimearg, help="Exclude all tweets from the given date and time. This is used for definition of a test data set.")

modelconfgroup = trainargparser.add_argument_group(title="Model Configuration", description="Selection and configuration of the network model.")
modelconfgroup.add_argument('--words', '-w', default=20000, type=int, help="Vocabulary size. Most common words are used, remaining are ignored.")
modelconfgroup.add_argument('--layers', '-l', default=1, type=int, help="Number of layers for models which support this setting (default: %(default)d)")
modelconfgroup.add_argument('--recurrent-layers', '-rl', default=1, type=int, help="Number of recurrent layers (default: %(default)d)")
modelconfgroup.add_argument('--conv-layers', '-cl', default=2, type=int, help="Number of convolutional/pooling layer pairs (default: %(default)d)")
modelconfgroup.add_argument('--units', '-u', default=64, type=int, help="Number of units per dense layer (default: %(default)d)")
modelconfgroup.add_argument('--embedding-units', '-eu', default=16, type=int, help="Number of units per dense layer (default: %(default)d)")
modelconfgroup.add_argument('--recurrent-units', '-ru', default=32, type=int, help="Number of units per recurrent layer (default: %(default)d)")
modelconfgroup.add_argument('--conv-filters', '-cf', default=32, type=int, help="Number of filters per convolutional layer (default: %(default)d)")
modelconfgroup.add_argument('--conv-window', '-cw', default=7, type=int, help="Convolutional window size (default: %(default)d)")
modelconfgroup.add_argument('--pool-size', '-P', default=2, type=int, help="Pooling window size (default: %(default)d)")
modelconfgroup.add_argument('--flatten-after-conv', '-f', action='store_true', help="Use a Flatten layer instead of GlocalMaxPool1D after last convolutional layer")
modelconfgroup.add_argument('--dropout', '-d', default=0.5, type=float, help="Dropout rate applied to input of dense and recurrent layers (default: %(default)0.1f).")
modelconfgroup.add_argument('--recurrent-dropout', '-dr', default=0.5, type=float, help="Dropout rate applied to recurrent units ()default: %(default)0.1f).")
modelconfgroup.add_argument('--dense-activation', '-a', default='relu', help="Activation function of dense layers (default: %(default)s)")
modelconfgroup.add_argument('--final-activation', '-fa', default='sigmoid', help="Activation function of last dense layer (default: %(default)s)")
modelconfgroup.add_argument('--recurrent-activation', '-ra', default='relu', help="Activation function of recurrent layers (default: %(default)s)")
modelconfgroup.add_argument('--conv-activation', '-ca', default='relu', help="Activation function of convolutional layers (default: %(default)s)")

modelselection = modelconfgroup.add_mutually_exclusive_group(required=True)
modelselection.add_argument('--dense', action='store_true', help="Simple dense network with number of units per layer defined by --units. Number of layers can be specified with --layers.")
modelselection.add_argument('--embedding-dense', action='store_true', help="Embedding layer (output dimension specified with --embedding-dimension) followed by dense layer(s) (count specified by --layers) with units per dense layer specified by --units.")
modelselection.add_argument('--gru', action='store_true', help="Word embedding layer (output dimension specified with --embedding-dimension) followed by one or multiple GRU layers (--recurrent-layers) with number of units specified by --recurrent-units. Finally, multiple dense layers (--layers, --units) may appear.")
modelselection.add_argument('--lstm', action='store_true', help="Word embedding layer (output dimension specified with --embedding-dimension) followed by one or multiple LSTM layers (--recurrent-layers) with number of units specified by --recurrent-units. Finally, multiple dense layers (--layers, --units) may appear.")
modelselection.add_argument('--convnet', action='store_true', help="Word embedding layer (output dimension specified with --embedding-dimension) followed by one or multiple 1D convolutional layers (--conv-layers) with number of filters specified by --conv-filters and a convolution window defined by --conv-window. The pooling size of pooling layers between convolutional layers is defined with --pool-size. Finally, multiple dense layers (--layers, --units) may appear.")

trainingconfgroup = trainargparser.add_argument_group(title="Training Configuration", description="Configuration of training parameters.")
trainingconfgroup.add_argument('--optimizer', '-O', default='rmsprop', help="Selection of optimizer algorithm (default: %(default)s)")
trainingconfgroup.add_argument('--loss', '-L', default='mse', help="Selection of loss function (default: %(default)s)")
trainingconfgroup.add_argument('--metric', '-M', default=['mae'], nargs='*', help="Selection of metrics (default: %(default)s)")
trainingconfgroup.add_argument('--epochs', '-e', default=10, type=int, help="Maximum number of training epochs (%(default)d). Training is aborted when overfitting appears. This can be disabled with --allow-overfitting.")
trainingconfgroup.add_argument('--early-stopping', '-E', action='store_true', help="Stop training when overfitting appears. Can be further configured with following parameters.")
trainingconfgroup.add_argument('--early-stopping-metric', '-S', default='val_loss', help="Metric monitored for early stopping of training to prevent overfitting.")
trainingconfgroup.add_argument('--patience', '-p', default=1, type=int, help="How many epochs the early stop metric may get worse until training is aborted (default: %(default)s)")
trainingconfgroup.add_argument('--batch-size', '-b', default=32, type=int, help="Training batch size")
validationgroup = trainingconfgroup.add_mutually_exclusive_group()
validationgroup.add_argument('--validation-split', '-s', type=float, help="Fraction of data set used for model validation while training")
validationgroup.add_argument('--k-fold', '-k', type=int, help="Do k-fold validation")

scoregroup = trainargparser.add_argument_group(title="Tweet Scoring", description="Tweet classification is based on a score between 0 and 1 (less interesting < more interesting). Training can distinguish scores between own tweets, retweets and favorites.")
scoregroup.add_argument('--score-owntweet', '-so', default=1.0, type=float, help="Score of own tweets. (%(default)0.1f)")
scoregroup.add_argument('--score-retweet', '-sr', default=1.0, type=float, help="Score of retweets. (%(default)0.1f)")
scoregroup.add_argument('--score-favorite', '-sf', default=1.0, type=float, help="Score of favorited tweets. (%(default)0.1f)")

predictargparser = subargparsers.add_parser('predict', help="Predict interesting Tweets with given trained model and Tweet database.")
predictargparser.add_argument('prefix', nargs=1, help="Prefix of model, tokenizer and state files stored previously by train command")
predictargparser.add_argument('--start-time', '-t', type=datetimearg, help="Predict all tweets from this date/time instead of last tweet used for training.")
predictargparser.add_argument('--min-score', '-s', type=float, default=0.8, help="Tweets scored with this or greater are interesting (default: %(default)1.1f)")

evaluateparser = subargparsers.add_parser('evaluate', help="Evaluate how well the predictions of a model perform against new or test datasets.")
evaluateparser.add_argument('prefix', nargs=1, help="Prefix of model, tokenizer and state files stored previously by train command")
evaluateparser.add_argument('--start-time', '-t', type=datetimearg, help="Predict all tweets from this date/time instead of last tweet used for training.")

args = argparser.parse_args()
if args.debug:
    logger.setLevel(logging.DEBUG)
logger.debug("Arguments: %s", str(args))

db = sqlite3.connect(args.database)
db.row_factory = sqlite3.Row
dbc = db.cursor()
dbc.execute("CREATE TABLE IF NOT EXISTS tweets (id INTEGER PRIMARY KEY, time DATETIME, user INTEGER, text TEXT, favorite BOOLEAN, retweet BOOLEAN, own BOOLEAN)")
db.commit()

if args.command is not None:
    globals()["cmd_%s" % args.command](args)
else:
    print("Missing command!", file=sys.stderr)
    argparser.print_usage()
    sys.exit(1)
