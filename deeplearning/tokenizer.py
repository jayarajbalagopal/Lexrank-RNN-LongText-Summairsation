import init
from nltk.tokenize import word_tokenize
from os import path
import config
import pickle
from parse_articles import description_parser


def tokenize_sentence(sentence):
    return ' '.join(list(
        word_tokenize(sentence)))

def article_is_complete(a):
    if ('head' not in a) or ('desc' not in a):
        return False
    if (a['head'] is None) or (a['desc'] is None):
        return False
    return True

def tokenize_articles(articles):
    tokenized = []
    N = len(articles)
    for i, a in enumerate(articles.values()):
        if article_is_complete(a):
            head = description_parser(a['desc'])
            tokenized.append((
                tokenize_sentence(a['head']),
                tokenize_sentence(head)))
    print('Tokenized {:,} / {:,} articles'.format(i+1, N))
    return tuple(map(list, zip(*tokenized)))

def pickle_articles(articles):
    with open(path.join(config.path_data, 'tokens.pkl'), 'wb') as f:
        pickle.dump(articles, f, 2)

def main():

    filename = path.join(config.path_articles,"news.jsonl")
    articles = init.load_articles(filename)
    final = tokenize_articles(articles)
    print(final)
    pickle_articles(final)


if __name__ == "__main__":
	main()
