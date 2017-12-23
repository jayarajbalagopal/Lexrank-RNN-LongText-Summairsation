from os import path

path_base = path.dirname(path.abspath(__file__))
path_data = path.join(path_base, 'data')
path_articles = path.join(path_base, 'articles')
path_glove = path.join(path_base, 'glove')
path_models = path.join(path_base, 'models')
path_logs = path.join(path_models, 'logs')
path_outputs = path.join(path_base, 'outputs')


# ****************************************** #
# Training config
# ****************************************** #

valid_size = 16  
valid_window = 100  
num_sampled = 64    

loss_eval_freq = 2000
sim_eval_freq = 50000
num_nearest_neigh = 8

max_plot_points = 500
tsne_iter = 5000
tsne_perplexity = 30