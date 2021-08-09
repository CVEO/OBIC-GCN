
import time
import numpy as np
import tensorflow as tf
from sklearn import svm, metrics
from sklearn import naive_bayes 
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from scipy import sparse
import sys


# from utils import load_data

# Set random seed



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'lsvm', 'Model string.')  # 'lsvm', 'rbfsvm', 'mlp', 'nb'
flags.DEFINE_string('npz', 'dd', 'Input npz data.')
# flags.DEFINE_string('logfile', 'ff', 'Output log file.')
flags.DEFINE_string('resfile', 'rr', 'Output result file.')
flags.DEFINE_string('mode', 'eval', 'Evaluate mode or predict mode.') 
flags.DEFINE_integer('pca', 0, 'Reduce dimensions (0 means no pca)')
flags.DEFINE_integer('svd', 0, 'Reduce dimensions (0 means no svd)')
flags.DEFINE_integer('seed', 12, 'random seed.')

seed = int(FLAGS.seed)
np.random.seed(seed)

t_start = time.time()

# data = np.load('data/ah/ah_testfull.npz')
data = np.load(str(FLAGS.npz), allow_pickle=True)
# data = np.load('data/ah/ah_subset_50.npz')
# data = np.load('data/ah/ah_subset_20.npz')
# data = np.load('data/ah/ah_subset_10.npz')
# data = np.load('data/ah/ah_subset_6.npz')
# data = np.load('data/ah/ah_subset_5.npz')

features = data['features'][()]
y_train = data['y_train'][()]
train_mask = data['train_mask'][()]
y_val = data['y_val'][()]
val_mask = data['val_mask'][()]
y_test = data['y_test'][()]
test_mask = data['test_mask'][()]

# 稀疏矩阵转正常矩阵
# features = features.toarray()

# LSA降维
# svd = TruncatedSVD(n_components=30, random_state=seed)
# print(features.shape)
# svd.fit(features) 
# features = svd.transform(features) 
# print(features.shape)
# print(y_train.shape)
# PCA 降维
if int(FLAGS.pca) != 0:
    pca = PCA(n_components=int(FLAGS.pca), random_state=seed)
    features = features.toarray()
    pca.fit(features) 
    features = sparse.lil_matrix(pca.transform(features) , dtype='float32')
if int(FLAGS.svd) > 0:
    svd = TruncatedSVD(n_components = FLAGS.svd, random_state=seed)
    svd.fit(features) 
    features = sparse.lil_matrix(svd.transform(features) , dtype='float32')

X = features[np.where(train_mask == True)]
y = y_train[np.where(train_mask == True)]
print(X.shape)
print(y.shape)

# X_val = features[np.where(val_mask == True)]
# y_val = y_val[np.where(val_mask == True)]
# print(X_val.shape)
# print(y_val.shape)

X_test = features[np.where(test_mask == True)]
y_test = y_test[np.where(test_mask == True)]
print(X_test.shape)
print(y_test.shape)

y = np.argmax(y, axis=1)
# y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)


if str(FLAGS.model) == 'lsvm':
    clf = svm.SVC(kernel='linear')
elif str(FLAGS.model) == 'rbfsvm':
    clf = svm.SVC(kernel='rbf')
elif str(FLAGS.model) == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, solver='adam', verbose=0, tol=1e-4, random_state=None, learning_rate='constant', learning_rate_init=.01)
elif str(FLAGS.model) == 'nb':
    clf = naive_bayes.GaussianNB()
    X = X.toarray()
    X_test = X_test.toarray()
elif str(FLAGS.model) == 'xgboost':
    clf = XGBClassifier(learning_rate=1,
        n_estimators=10, # 树的个数--1000棵树建立xgboost
        max_depth=3, # 树的深度
        min_child_weight = 1, # 叶子节点最小权重
        gamma=0., # 惩罚项中叶子结点个数前的参数
        subsample=0.8, # 随机选择80%样本建立决策树
        colsample_btree=0.8, # 随机选择80%特征建立决策树
        objective='multi:softmax', # 指定损失函数
        # scale_pos_weight=1, # 解决样本个数不平衡的问题
        random_state=seed # 随机数
        ) 
    X = X.toarray()
    X_test = X_test.toarray()
elif str(FLAGS.model) == 'randomforest':
    clf = RandomForestClassifier()
    X = X.toarray()
    X_test = X_test.toarray()

clf.fit(X, y)

y_pred = clf.predict(X_test)

total_time = time.time() - t_start

# Evalute_BOOL = True
# Predict_BOOL = False

with open(str(FLAGS.resfile), 'a') as f:
    if str(FLAGS.mode) == 'eval':
        # print(metrics.classification_report(y_test, y_pred, digits=5))
        print("OA : {:0.5f}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Kappa : {:0.5f}".format(metrics.cohen_kappa_score(y_test, y_pred)))
        print("F1 : {:0.5f}".format(metrics.f1_score(y_test, y_pred, average='weighted')))
        print("total time:", "{:.5f}s".format(total_time))
        # result_file.write(date_now + '\n')
        f.write(' '.join(sys.argv) + '\n')
        f.write("OA : {:0.5f} ".format(metrics.accuracy_score(y_test, y_pred)))
        f.write("Kappa : {:0.5f} ".format(metrics.cohen_kappa_score(y_test, y_pred)))
        f.write("F1 : {:0.5f} ".format(metrics.f1_score(y_test, y_pred, average='weighted')))
        f.write("total time: {:.5f}s \n".format(total_time))

    elif str(FLAGS.mode) == 'pred':
        # result_file = 'result_ah_pred_svm.log'
        for i in range(y_pred.shape[0]):
            f.write(str(y_pred[i]) + '\n')
f.close()
