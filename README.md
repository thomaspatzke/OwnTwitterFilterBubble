# Build your Own Twitter Filter Bubble
...with tweets you liked and (re)tweeted!

## Introduction

This is one of my first deep learning experiments. Basically, it tries to learn your interests under the assumption that
you like and retweet Tweets that match your interests and that you are interested in such things you're tweeting about
yourself. This is accomplished by assigning a score between 0 and 1 to each Tweet depending on the above factors and use
this as training input for various neural networks. The following networks are included in the code:

* Very simple dense-only network
* Embedding input layer (also used for all following networks)
* Recurrent GRU layers
* Recurrent LSTM layers
* Convolutional layers

All networks end with dense classifier layer(s). After successful training, the model predicts scores for new tweets.

This project was implemented with Python 3 and the [Keras](https://keras.io) deep learning library.

## Installation

Python3 with TK support (`apt-get install python3-tk`) is required. Further, the dependencies from *requirements.txt*
must be installed with:

```
pip3 install -r requirements.txt
```

## Usage

There are three modes of operations:

* Tweet aquisition with the *gettweets* subcommand.
* Training and validation with the *train* subcommand.
* Prediction with the *predict* subcommand.
* Evaluation how well the predictions are with the *evaluate* subcommand.

### Getting Tweets

A Twitter API key must be obtained and configured in a file. This should run regularly for weeks or even months, as you
need a big amount of Tweets to get reasonable results. I recommend a minimum of 10.000.

The API key should be put in a file *twitter.conf* (default name):

```
--consumer-key ...
--consumer-secret ...
--access-token ...
--access-token-secret ...
```

The following invocation gets the latest tweets from your time line and stores them in a local SQLite database called
*tweets.db*:
```
./otfb.py gettweets

```

As your favorites, tweets and retweets are quite important for training of the networks, you should get your history of
these with:

```
./otfb.py gettweets --all

```

Nevertheless, if you're an inactive Tweeter, your training set could be imbalanced which could cause inaccurate results.
Currently, no countermeasures are implemented against this.

### Building, Evaluating and Training of Networks

After you obtained a sufficient number of Tweets, you can go ahead and play around with the networks. There is a big
number of parameters that allow to adjust the configuration of networks and training process. Finally, models can be
saved with `-o`. A saved model consists of three files (weights, tokenizer and some useful metadata), therefore only a
file name prefix is expected.

An introduction to deep learning and methodologies is not in scope of this documentation. I recommend to read the really
good book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) from [François
Chollet](https://twitter.com/fchollet) or one of the free tutorials, [like this
one](https://machinelearningmastery.com/start-here/).

### Predict Tweets

Run the following command to predict interesting Tweets:

```
./otfb.py predict <model prefix>
```

By default, all tweets after the trained set are predicted. You can set a prediction start time with `-t` and a minimum
score with `-s`.

If you use *evaluate* instead of *predict* subcommand, the predictions are evaluated against your real interaction and
some metrics are calculated.

## Results

First experiments with slightly more than 11.000 tweets and a 60:40 ratio of uninteresting to interesting tweets
resulted in quite good predictions:

* Many models performed with mean absolute errors less than 0.05 on test sets not seen before.
* Comparing predicted tweets with these I've seen on the same day, the predicted set contained many tweets I've considered
  interesting.
* Surprisingly, the simple and performant *embedding-dense* model seems to perform better than other models with more
  complex layers.

These are first impressions. I plan to write a blog post with results of longer observations.

## Disclaimer

This is experimental code. Output generated by it may not be compatible with future versions of it. I will
intentionally break things if this is required for further experiments and will not even try to be backwards-compatible
or build converters.

Further, be aware that I'm just a beginner in this area. I've just read few books and a bunch of blog posts on this
topic.

Even if it's your very personal custom filter bubble, don't stay too long in it.

## Contact

Feedback would be much appreciated, especially such feedback that identifies errors and improvements :-)

Feel free to contact me [by mail](mailto:thomas@patzke.org), [Twitter](https://twitter.com/blubbfiction) or other means
of communication of your choice.

## License

This code is released under LGPLv3.
