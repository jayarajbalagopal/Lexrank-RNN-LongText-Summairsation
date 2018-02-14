import config
import json
import pickle
import numpy as np
from os import path
from type import DataContainer,ArticleContainer

n_to_load = 1000 #no of articles to be loaded from json file

def get_train_val_test_keys(keys, val_pct=.1, test_pct=.1):

    n = keys.shape[0]
    np.random.shuffle(keys)
    test_cutoff = 1 - test_pct
    val_cutoff = test_cutoff - val_pct
    return np.split(keys, [int(val_cutoff * n), int(test_cutoff * n)])

def load_article_container(filename_pickle):

    with open(filename_pickle + '_train.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(filename_pickle + '_validation.pkl', 'rb') as f:
        validation = pickle.load(f)
    with open(filename_pickle + '_test.pkl', 'rb') as f:
        test = pickle.load(f)
    return DataContainer(train, validation, test)

def files_to_containers(files,article):

    heads = np.array([article[f]['head'] for f in files])
    desc = np.array([article[f]['desc'] for f in files])
    return ArticleContainer(heads,desc)

def save_data_container(data, filename_pickle):

    with open(filename_pickle + '_train.pkl', 'wb') as f:
        pickle.dump(data.train, f)
    with open(filename_pickle + '_validation.pkl', 'wb') as f:
        pickle.dump(data.validation, f)
    with open(filename_pickle + '_test.pkl', 'wb') as f:
        pickle.dump(data.test, f)
    print('Data container saved')


def pickled_data_container_exists(filename_pickle):
    
    if not path.exists(filename_pickle + '_train.pkl'):
        return False
    elif not path.exists(filename_pickle + '_validation.pkl'):
        return False
    elif not path.exists(filename_pickle + '_test.pkl'):
        return False
    else:
        return True


def load_articles(filename):

	articles = dict()
	c = 0
	with open(filename, 'r') as f:
		for l in f:
			news = json.loads(l)
			articles[news['id']] = {"head" : news['title'],
									"desc" : news['content']
									}
			c = c+1
			if c>=n_to_load:
				break

	print("Loaded {} articles from json file".format(c))
	return articles


def save_articles(filename_pickle):
	
	filename = path.join(config.path_articles,"news.jsonl")
	articles = load_articles(filename)

	files = np.array([art for art in articles])
	train_files, validation_files, test_files = get_train_val_test_keys(files)
	print('Data split into segments of size {:,} (train), {:,} (validation), and {:,} (test)'.format(
		train_files.shape[0], validation_files.shape[0], test_files.shape[0]))

	files_to_containers(test_files, articles)	
	data = DataContainer(
		files_to_containers(train_files, articles),
		files_to_containers(validation_files, articles),
		files_to_containers(test_files, articles),
		)
	save_data_container(data, filename_pickle)

	return data


def main():
	filename_pickle = path.join(config.path_data, 'data_processed')
	if not pickled_data_container_exists(filename_pickle):
		data = save_articles(filename_pickle)
	else:
		print('Loading pickled data')
		data = load_article_container(filename_pickle)
	return data

if __name__ == "__main__":
	main()