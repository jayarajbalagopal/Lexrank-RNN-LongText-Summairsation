from collections import namedtuple


ArticleContainer = namedtuple('ArticleContainer', ['head', 'desc'])
DataContainer = namedtuple('DataContainer', ['train', 'validation', 'test'])