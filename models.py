"""Architecture ported from the uploaded paper-rineng.ipynb.
TensorFlow is imported only when a model is built. No training on import.
"""

def psd_mlp(input_dim=80, lr=5e-4):
    import tensorflow as tf
    L=tf.keras.layers; reg=tf.keras.regularizers
    inp=L.Input(shape=(input_dim,),name='psd_input')
    x=L.Dense(64,kernel_regularizer=reg.l2(1e-4),name='dense_64')(inp)
    x=L.BatchNormalization(name='bn_64')(x);x=L.Activation('relu',name='relu_64')(x);x=L.Dropout(.35,name='dropout_64')(x)
    x=L.Dense(32,kernel_regularizer=reg.l2(1e-4),name='dense_32')(x)
    x=L.BatchNormalization(name='bn_32')(x);x=L.Activation('relu',name='relu_32')(x);x=L.Dropout(.25,name='dropout_32')(x)
    x=L.Dense(16,kernel_regularizer=reg.l2(1e-4),name='dense_16')(x)
    x=L.Activation('relu',name='relu_16')(x)
    out=L.Dense(1,activation='sigmoid',name='p_asd')(x)
    m=tf.keras.Model(inp,out,name='PSD_Lightweight_MLP')
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return m


def cwt_cnn(input_shape=(64,128,16),lr=1e-5):
    import tensorflow as tf
    L=tf.keras.layers;reg=tf.keras.regularizers
    inp=L.Input(shape=input_shape,name='cwt_input')
    x=L.Conv2D(24,3,padding='same',use_bias=False,kernel_regularizer=reg.l2(1e-4),name='conv_initial')(inp)
    x=L.BatchNormalization(name='bn_initial')(x);x=L.Activation('relu',name='act_initial')(x)
    x=L.MaxPooling2D(2,name='pool_initial')(x);x=L.Dropout(.30,name='drop_initial')(x)
    for filters,drop in [(48,.25),(96,.30),(128,None)]:
        x=L.SeparableConv2D(filters,3,padding='same',use_bias=False,
              depthwise_regularizer=reg.l2(1e-4),pointwise_regularizer=reg.l2(1e-4),name=f'sepconv_{filters}')(x)
        x=L.BatchNormalization(name=f'bn_{filters}')(x)
        activation={48:'act_48',96:'cwt_last_conv_act',128:'cwt_last_conv_act_2'}[filters]
        x=L.Activation('relu',name=activation)(x)
        if filters!=128:
            x=L.MaxPooling2D(2,name=f'pool_{filters}')(x)
            x=L.Dropout(drop,name=f'drop_{filters}')(x)
    x=L.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x=L.Dense(64,activation='relu',kernel_regularizer=reg.l2(1e-4),name='dense_64')(x)
    x=L.Dropout(.30,name='drop_dense')(x)
    out=L.Dense(1,activation='sigmoid',name='p_asd')(x)
    m=tf.keras.Model(inp,out,name='CWT_Depthwise_Separable_CNN')
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return m


def graphconv_class():
    import tensorflow as tf
    L=tf.keras.layers
    @tf.keras.utils.register_keras_serializable(package='GraphLayers')
    class GraphConv(L.Layer):
        def __init__(self,units,activation=None,use_bias=True,kernel_regularizer=None,**kwargs):
            super().__init__(**kwargs)
            self.units=int(units);self.activation=tf.keras.activations.get(activation)
            self.use_bias=bool(use_bias)
            self.kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer)
        def build(self,input_shape):
            self.kernel=self.add_weight(name='kernel',shape=(int(input_shape[0][-1]),self.units),
                 initializer='glorot_uniform',regularizer=self.kernel_regularizer,trainable=True)
            self.bias=self.add_weight(name='bias',shape=(self.units,),initializer='zeros',trainable=True) if self.use_bias else None
            super().build(input_shape)
        def call(self,inputs):
            x,adj=inputs
            h=tf.linalg.matmul(tf.cast(adj,tf.float32),tf.cast(x,tf.float32))
            h=tf.linalg.matmul(h,self.kernel)
            if self.bias is not None:h=h+self.bias
            if self.activation is not None:h=self.activation(h)
            return h
        def get_config(self):
            d=super().get_config(); d.update(units=self.units,
              activation=tf.keras.activations.serialize(self.activation),use_bias=self.use_bias,
              kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer))
            return d
    return GraphConv


def edge_aware_gcn(node_shape=(16,5),adj_shape=(16,16),edge_shape=(120,),
                   lr=3e-4,weight_decay=1e-4,dropout_rate=.35):
    import tensorflow as tf
    L=tf.keras.layers;reg=tf.keras.regularizers; GraphConv=graphconv_class()
    ni=L.Input(shape=node_shape,name='node_features_input')
    ai=L.Input(shape=adj_shape,name='adjacency_input')
    ei=L.Input(shape=edge_shape,name='edge_features_input')
    def block(x,units,name):
        h=GraphConv(units,activation=None,use_bias=False,kernel_regularizer=reg.l2(weight_decay),name=f'{name}_gconv_1')([x,ai])
        h=L.BatchNormalization(name=f'{name}_bn_1')(h)
        h=L.Activation('swish',name=f'{name}_swish_1')(h)
        h=L.Dropout(dropout_rate,name=f'{name}_drop_1')(h)
        h=GraphConv(units,activation=None,use_bias=False,kernel_regularizer=reg.l2(weight_decay),name=f'{name}_gconv_2')([h,ai])
        h=L.BatchNormalization(name=f'{name}_bn_2')(h)
        if int(x.shape[-1])!=units:
            skip=L.Dense(units,use_bias=False,kernel_regularizer=reg.l2(weight_decay),name=f'{name}_skip_projection')(x)
            skip=L.BatchNormalization(name=f'{name}_skip_bn')(skip)
        else:skip=x
        h=L.Add(name=f'{name}_residual_add')([skip,h])
        h=L.Activation('swish',name=f'{name}_swish_out')(h)
        return L.Dropout(dropout_rate,name=f'{name}_drop_out')(h)
    x=block(ni,32,'gcn_block_1');x=block(x,48,'gcn_block_2');x=block(x,64,'gcn_block_3')
    g=L.Concatenate(name='graph_pool_concat')([
       L.GlobalAveragePooling1D(name='graph_global_mean_pool')(x),
       L.GlobalMaxPooling1D(name='graph_global_max_pool')(x)])
    g=L.Dense(96,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='graph_dense_96')(g)
    g=L.BatchNormalization(name='graph_bn_96')(g);g=L.Dropout(dropout_rate,name='graph_dropout_96')(g)
    e=L.Dense(128,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='edge_dense_128')(ei)
    e=L.BatchNormalization(name='edge_bn_128')(e);e=L.Dropout(dropout_rate,name='edge_dropout_128')(e)
    e=L.Dense(64,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='edge_dense_64')(e)
    e=L.BatchNormalization(name='edge_bn_64')(e);e=L.Dropout(dropout_rate,name='edge_dropout_64')(e)
    nf=L.Flatten(name='node_feature_flatten')(ni)
    nf=L.Dense(64,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='node_dense_64')(nf)
    nf=L.BatchNormalization(name='node_bn_64')(nf);nf=L.Dropout(dropout_rate,name='node_dropout_64')(nf)
    z=L.Concatenate(name='multibranch_concat')([g,e,nf])
    z=L.Dense(128,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='fusion_dense_128')(z)
    z=L.BatchNormalization(name='fusion_bn_128')(z);z=L.Dropout(dropout_rate,name='fusion_dropout_128')(z)
    z=L.Dense(64,activation='swish',kernel_regularizer=reg.l2(weight_decay),name='fusion_dense_64')(z)
    z=L.BatchNormalization(name='fusion_bn_64')(z);z=L.Dropout(dropout_rate,name='fusion_dropout_64')(z)
    out=L.Dense(1,activation='sigmoid',name='p_asd')(z)
    m=tf.keras.Model(inputs=[ni,ai,ei],outputs=out,name='EdgeAware_Lightweight_GCN')
    try:opt=tf.keras.optimizers.AdamW(learning_rate=lr,weight_decay=weight_decay)
    except AttributeError:opt=tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return m
