from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D, Conv2D
from keras.models import Model

def dense_model(Inputs,nclasses,nregclasses,dropoutRate=0.25):
    """
    Dense matrix, defaults similat to 2016 training
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Flatten()(Inputs[0])
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)#Inputs[0])
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model(Inputs):
    """
    One hidden layer model
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform')(Inputs)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def linear_model(Inputs):
    """
    Linear model
    """
    #  Here add e.g. the normal dense stuff from DeepCSV
    predictions = Dense(1, activation='linear',kernel_initializer='lecun_uniform')(Inputs)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
