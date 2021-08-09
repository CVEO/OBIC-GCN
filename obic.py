from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import metrics
from scipy import sparse

from gcn.utils import *
from gcn.models import GCN, MLP

from utils import load_data





###
# python obic.py --model gcn_cheby --nodes 64 --layers 1 --dropout 0.3 --max_degree 3 --svd 20
###

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('npz', 'dd', 'Input npz data.')
flags.DEFINE_string('logfile', 'ff', 'Output log file.')
flags.DEFINE_string('resfile', 'rr', 'Output result file.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('nodes', 32, 'Number of units in hidden layer.')
flags.DEFINE_integer('layers', 1, 'Number of hidden layers.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('svd', 0, 'Reduce dimensions (0 means no svd)')
flags.DEFINE_integer('seed', 123, 'random seed.')

# Set random seed
# seed = 123
np.random.seed(int(FLAGS.seed))
tf.set_random_seed(int(FLAGS.seed))

# log & result file
date_now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

log_file = open(str(FLAGS.logfile), 'a')
log_file.write(date_now + '\n')
log_file.write(' '.join(sys.argv) + '\n')

result_file = open(str(FLAGS.resfile), 'a')
result_file.write(date_now + '\n')
result_file.write(' '.join(sys.argv) + '\n')

# load data
# adj, features, y_train, train_mask, y_val, val_mask = load_data("data/tj/lzw.tif", "data/tj/features.txt", "data/tj/train.txt", "data/tj/test.txt")

# data = np.load('data/ah/ah_subset_50.npz')
# data = np.load('data/ah/ah_subset_20.npz')
# data = np.load('data/ah/ah_subset_10.npz')
# data = np.load('data/ah/ah_subset_6.npz')
# data = np.load('data/ah/ah_subset_5.npz')

# data = np.load('data/fj/fj.npz', allow_pickle=True)
# data = np.load('data/ah/ah_1_1_1_1.npz', allow_pickle=True)
data = np.load(str(FLAGS.npz), allow_pickle=True)

adj = data['adj'][()]
features = data['features'][()]
y_train = data['y_train'][()]
train_mask = data['train_mask'][()]
y_val = data['y_val'][()]
val_mask = data['val_mask'][()]
y_test = data['y_test'][()]
test_mask = data['test_mask'][()]

if FLAGS.svd > 0:
    svd = TruncatedSVD(n_components = FLAGS.svd, random_state=int(FLAGS.seed))
    svd.fit(features) 
    features = sparse.lil_matrix(svd.transform(features) , dtype='float32')


# pca = PCA(n_components=20, random_state=seed)
# features = features.toarray()
# pca.fit(features) 
# features = sparse.lil_matrix(pca.transform(features) , dtype='float32')

# print(repr(adj))
# print(repr(features))
# print(repr(y_train))
# print(repr(train_mask))
# print(y_val.shape)
# print(repr(val_mask))
# print(y_test.shape)
# exit()

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
best_voa = 0.
t_start = time.time()

for epoch in range(FLAGS.epochs):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    log_file.write('{} {} {} {} {} {} {} {} {} {}\n'.format("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc)))

    if acc > best_voa:
        best_voa = acc

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

total_time = time.time() - t_start

pred = sess.run(tf.argmax(model.outputs, 1), construct_feed_dict(features, support, y_test, test_mask, placeholders))
y_test = np.argmax(y_test, axis=1)
test_mask = test_mask.astype('float')

oa = metrics.accuracy_score(y_test, pred, sample_weight=test_mask)
kappa = metrics.cohen_kappa_score(y_test, pred, sample_weight=test_mask)
fscore = metrics.f1_score(y_test, pred, average='weighted', sample_weight=test_mask)

print("Optimization Finished!, best VOA:", best_voa, "OA:", oa, 'Kappa', kappa, 'F1-score', fscore, "total time:", "{:.5f}s".format(total_time))

# log 
log_file.write('\n\n\n')
log_file.close()

result_file.write('best VOA:\t{}\tOA:\t{}\tKappa\t{}\tF1-score\t{}\ttotal time:\t{}\n\n'.format(best_voa, oa, kappa, fscore, "{:.5f}".format(total_time)))
result_file.close()