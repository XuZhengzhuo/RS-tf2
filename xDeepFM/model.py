"""
Created on August 20, 2020

model: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input


class DNN(layers.Layer):
    """
    DNN part
    """
    def __init__(self, hidden_units, dnn_dropout=0., dnn_activation='relu'):
        """
        DNN为 n 层线性层，论文里也说是 plain DNN
        :param hidden_units: A list. list of hidden layer units's numbers.
        :param dnn_dropout: A scalar. dropout number.
        :param dnn_activation: A string. activation function.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class Linear(layers.Layer):
    """
        线性层
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class CIN(layers.Layer):
    """
    CIN part 核心内容
    """
    def __init__(self, cin_size, l2_reg=1e-4):
        """
        :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
        :param l2_reg: A scalar. L2 regularization.
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # 构建CIN
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters 可以看出这里没有共享参数
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        # 这个变量保存每一层的结果
        # 保证第一层是自己和自己的外积
        hidden_layers_results = [inputs] 
        # split dimension 2 for convenient calculation
        # dim * (None, field_nums[0], 1)
        # 在第3个维度上分两份
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        # 迭代实现每一层的计算 注意 cin_size是个list 值是H_k
        for idx, size in enumerate(self.cin_size):
            # dim * (None, filed_nums[i], 1)
            # 计算 x_k X_k是
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  
            # (dim, None, field_nums[0], field_nums[i])
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  
            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])
            # (None, dim, field_nums[0] * field_nums[i])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2]) 
            # 这一步是论文里的公式6
            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')
            # (None, field_num[i+1], dim)
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  
            # 这一步得到了x_K
            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        # (None, H_1 + ... + H_K, dim)
        result = tf.concat(final_results, axis=1)
        # (None, dim)
        result = tf.reduce_sum(result,  axis=-1)  

        return result


class xDeepFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-5, cin_reg=1e-5):
        """
        xDeepFM
        :param feature_columns: A list. a list containing dense and sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. the regularizer of embedding.
        :param cin_reg: A scalar. the regularizer of cin.
        """
        super(xDeepFM, self).__init__()
        # 读入特征 包括稠密特征和稀疏特征
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # 稀疏特征的嵌入维度 默认 8
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # 线性层
        self.linear = Linear()
        # CIN模块
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        # DNN模块
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        # CIN的输出层
        self.cin_dense = Dense(1)
        # DNN的输出层
        self.dnn_dense = Dense(1)
        # 共享的偏差
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        # 首先分开稀疏和稠密特征
        dense_inputs, sparse_inputs = inputs
        # 对特征做嵌入
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # cin部分
        # 此处需要一步三维矩阵转置，0和1维度互换
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (None, embedding_nums, dim)
        cin_out = self.cin_dense(cin_out)
        # dnn部分
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)
        # sigmoid激活转化为点击率
        output = tf.nn.sigmoid(cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        # 记录模型结构
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()