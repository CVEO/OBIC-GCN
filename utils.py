import random
import numpy as np
from scipy import sparse
# from sklearn.decomposition import TruncatedSVD, PCA
# from sklearn.metrics.pairwise import cosine_similarity
# import networkx as nx


# Auhui Map
category_map = {
    'building': [1, 0, 0, 0, 0],
    'bare': [0, 1, 0, 0, 0],
    'road': [0, 0, 1, 0, 0],
    'vegetation': [0, 0, 0, 1, 0],
    'water': [0, 0, 0, 0, 1]
}

# Fujian Map
# category_map = {
#     'farmland': [1, 0, 0, 0, 0, 0, 0, 0] ,
#     'garden':   [0, 1, 0, 0, 0, 0, 0, 0] ,
#     'woodland': [0, 0, 1, 0, 0, 0, 0, 0] ,
#     'grass':    [0, 0, 0, 1, 0, 0, 0, 0] ,
#     'building': [0, 0, 0, 0, 1, 0, 0, 0] ,
#     'artifact': [0, 0, 0, 0, 0, 1, 0, 0] ,
#     'bareland': [0, 0, 0, 0, 0, 0, 1, 0] ,
#     'waters':   [0, 0, 0, 0, 0, 0, 0, 1]
# }

def load_data(mask_path, features_path, training_data_path, validation_data_path, test_data_path):
    

    from osgeo import gdal, gdal_array
    from gdalconst import GA_ReadOnly
    src_ds = gdal.Open(mask_path, GA_ReadOnly)
    objects_img = (gdal_array.DatasetReadAsArray(src_ds))
    count = np.max(objects_img, axis=None) + 1

    # features
    features_dim = 181
    features = np.zeros((count, features_dim), 'float32')
    with open(features_path) as file:
        for line in file.readlines():
            splited_line = line.split('\t')
            object_id = int(splited_line[0])
            features[object_id][:] = [float(feature) for feature in splited_line[1:]]
    features = sparse.lil_matrix(features, dtype='float32')

    # 1ordOBJ adjacency.
    # Adjacency from related_obj
    ###         BUGS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    adjency = np.zeros((count, count), dtype='float32')
    # adjency = sparse.lil_matrix((count, count), dtype='float32')
    for row_id, row in enumerate(objects_img):
        if (row_id+1)<src_ds.RasterYSize:
            for col_id, dn in enumerate(row):
                if (col_id+1)<src_ds.RasterXSize:
                    right_dn = objects_img[row_id, col_id + 1]
                    down_dn = objects_img[row_id + 1, col_id]
                    right_down_dn = objects_img[row_id + 1, col_id + 1]
                    if dn != right_dn:
                        adjency[dn, right_dn] = adjency[right_dn, dn] = 1
                    if dn != down_dn:
                        adjency[dn, down_dn] = adjency[down_dn, dn] = 1
                    if dn != right_down_dn:
                        adjency[dn, right_down_dn] = adjency[right_down_dn, dn] = 1

                if (col_id>0):
                    left_down_dn = objects_img[row_id + 1, col_id - 1]
                    if dn != down_dn:
                        adjency[dn, left_down_dn] = adjency[left_down_dn, dn] = 1
    
    out_adj = sparse.lil_matrix(adjency, dtype='float32')

    
    '''
    Train,val,test Masks.
    '''
    # train data
    # category_dim = 5
    category_dim = len(category_map)
    y_train = np.zeros((count, category_dim))
    train_mask = [False] * count
    
    with open(training_data_path) as file:
        for line in file.readlines():
            splited_line = line.split('\n')[0].split('\t')
            object_id = int(splited_line[0])
            category = category_map[splited_line[1]]
            y_train[object_id] = np.array(category, dtype='int')
            train_mask[object_id] = True
            
    train_mask = np.array(train_mask, dtype='bool')

    # validation data
    y_val = np.zeros((count, category_dim))
    val_mask = [False] * count
    
    with open(validation_data_path) as file:
        for line in file.readlines():
            splited_line = line.split('\n')[0].split('\t')
            object_id = int(splited_line[0])
            category = category_map[splited_line[1]]
            y_val[object_id] = np.array(category, dtype='int')
            val_mask[object_id] = True
    val_mask = np.array(val_mask, dtype='bool')

    # test data
    y_test = np.zeros((count, category_dim))
    test_mask = [False] * count
    
    with open(test_data_path) as file:
        for line in file.readlines():
            splited_line = line.split('\n')[0].split('\t')
            object_id = int(splited_line[0])
            category = category_map[splited_line[1]]
            y_test[object_id] = np.array(category, dtype='int')
            test_mask[object_id] = True
    test_mask = np.array(test_mask, dtype='bool')
    
    # all test.
    # y_test = np.zeros((28452, category_dim))
    # test_mask = [True] * 28452
    
    # with open(test_data_path) as file:
    #     for line in file.readlines():
    #         splited_line = line.split('\n')[0].split('\t')
    #         object_id = int(splited_line[0])
    #         category = category_map[splited_line[1]]
    #         if adj_1ordOBJ_BOOL:
    #             y_test[object_id] = np.array(category, dtype='int')
    #             test_mask[object_id] = True
    #         elif adj_cosSIM_BOOL:
    #             y_test[conc_feature_list.index(object_id)] = np.array(category, dtype='int')
    #             test_mask[conc_feature_list.index(object_id)] = True
    # test_mask = np.array(test_mask, dtype='bool')

    return out_adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask


def generate_npz(path):
    adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask = load_data("data/ah/ah_mask.tif", "data/ah/features.txt", "data/ah/train.txt", "data/ah/val.txt", "data/ah/test.txt")
    # adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask = load_data("data/fj/fj_mask.tif", "data/fj/features.txt", "data/fj/train.txt", "data/fj/val.txt", "data/fj/test.txt")
    np.savez(path, adj=adj, features=features, y_train=y_train, train_mask=train_mask, y_val=y_val, val_mask=val_mask, y_test=y_test, test_mask=test_mask)


def generate_sub_npz(input_path, output_path, sub_count):
    data = np.load(input_path)
    adj = data['adj'][()]
    features = data['features'][()]
    y_train = data['y_train'][()]
    train_mask = data['train_mask'][()]
    y_val = data['y_val'][()]
    val_mask = data['val_mask'][()]

    train_ids = np.where(train_mask == True)[0]    
    train_count = train_ids.shape[0]
    sub_ids = random.sample(range(train_count), sub_count)
    sub_train_ids = train_ids[sub_ids]

    print(sub_train_ids)

    # y_train = y_train[sub_train_ids]
    train_mask[:] = False
    train_mask[sub_train_ids] = True

    np.savez(output_path, adj=adj, features=features, y_train=y_train, train_mask=train_mask, y_val=y_val, val_mask=val_mask)



if __name__ == '__main__':

    # generate_npz('data/ah/ah_1_1_1_1.npz')
    generate_npz('data/test.npz')
    # generate_npz('data/fj/fj.npz')

    # data = np.load('data/ah/ah_1_1_1_1.npz', allow_pickle=True)    
    data = np.load('data/test.npz', allow_pickle=True)    
    # data = np.load('data/fj/fj.npz', allow_pickle=True)    
    adj = data['adj'][()]
    features = data['features'][()]
    y_train = data['y_train'][()]
    train_mask = data['train_mask'][()]
    y_val = data['y_val'][()]
    val_mask = data['val_mask'][()]
    y_test = data['y_test'][()]
    test_mask = data['test_mask'][()]

    print (repr(adj))
    print (repr(features))
    print (repr(y_train))
    print (repr(train_mask))
    print (repr(y_val))
    print (repr(val_mask))
    print (repr(y_test))
    print (repr(test_mask))
    # print(features)

