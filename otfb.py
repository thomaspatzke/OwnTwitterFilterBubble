#!/usr/bin/env python3
# Your Own Twitter Filter Bubble
# Learn your interests from your Twitter favorites, tweets and retweets and
# filter/reorder the timeline accordingly.

import sys
import argparse
import sqlite3
import tweepy
import logging
import itertools

logging.basicConfig()
logger = logging.getLogger()

tweepy_api_instance = None

### Tools ###
# Following class is copied from https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

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

### Commands ###
def cmd_gettweets():
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

### Main program ###
argparser = argparse.ArgumentParser(description="Your Own Twitter Filter Bubble")
argparser.add_argument("--database", "-d", default='tweets.db', help="SQLite database used internally to store Tweets. Will be created if not already existing. Default: %(default)s")
argparser.add_argument('--config', '-c', type=open, action=LoadFromFile, help="Configuration file with authentication data")
argparser.add_argument('--consumer-key', help="Twitter API Consumer Key (not recommended - sensible data may be stored in shell history!)")
argparser.add_argument('--consumer-secret', help="Twitter API Consumer Secret (not recommended - sensible data may be stored in shell history!)")
argparser.add_argument('--access-token', help="Twitter API Access Token (not recommended - sensible data may be stored in shell history!)")
argparser.add_argument('--access-token-secret', help="Twitter API Access Token Secret (not recommended - sensible data may be stored in shell history!)")
argparser.add_argument('--fetchcount', '-n', type=int, default=200, help="Number of tweets fetched per API request (Default: %(default)d)")
argparser.add_argument('--max-timeline-tweets', '-T', type=int, default=25000, help="Maximum number of fetched tweets (default: %(default)d)")
argparser.add_argument('--max-favorites', '-F', type=int, default=5000, help="Maximum number of fetched favorites (default: %(default)d)")
argparser.add_argument('--max-own-tweets', '-O', type=int, default=5000, help="Maximum number of fetched tweets of authenticated user (default: %(default)d)")
argparser.add_argument('--debug', '-D', action='store_true', help="Enable debug output")
subargparsers = argparser.add_subparsers(dest='command', help="Commands")

tweetargparser = subargparsers.add_parser('gettweets', help="Get tweets. By default, get timeline beginning with Tweet ID from last run.")
tweetargparser.add_argument('--favorites', '-f', action='store_true', help="Get user accounts favorites instead of timeline. Does not store tweets which are already stored in database.")
tweetargparser.add_argument('--own', '-o', action='store_true', help="Get user accounts tweets instead of timeline. Does not store tweets which are already stored in database.")
tweetargparser.add_argument('--all', '-a', action='store_true', help="Shortcut for -fro - get favorites and own (re)tweets instead of timeline and store missing tweets in database.")

args = argparser.parse_args()
if args.debug:
    logger.setLevel(logging.DEBUG)
logger.debug("Arguments: %s", str(args))

db = sqlite3.connect(args.database)
dbc = db.cursor()
dbc.execute("CREATE TABLE IF NOT EXISTS tweets (id INTEGER PRIMARY KEY, time DATETIME, user INTEGER, text TEXT, favorite BOOLEAN, retweet BOOLEAN, own BOOLEAN)")
db.commit()

if args.command is not None:
    globals()["cmd_%s" % args.command]()
else:
    print("Missing command!", file=sys.stderr)
    argparser.print_usage()
    sys.exit(1)
