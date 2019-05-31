from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras import backend as K

from utils import get_logger

logger = get_logger('model')


def build_model(dense_act='relu', kernel_initializer='he_uniform'):
    K.clear_session()
    # build model =====================================================================================
    # numerics
    numerics_input = Input(shape=(4,), name='numerics_input')
    numerics_dense = Dense(16, activation=dense_act, kernel_initializer=kernel_initializer)(numerics_input)
    numerics_dense = BatchNormalization()(numerics_dense)

    # prices
    prices_input = Input(shape=(50, ), name='prices_input')
    prices_dense = Dense(units=32, activation=dense_act, kernel_initializer=kernel_initializer,
                         name='merged_tcn_dense')(prices_input)
    prices_dense = BatchNormalization()(prices_dense)

    # clicks
    clicks_input = Input(shape=(100, ), name='clicks_input')
    clicks_dense = Dense(units=64, activation=dense_act, kernel_initializer=kernel_initializer,
                           name='cfilter_dense1')(clicks_input)
    cfilter_dense = BatchNormalization()(clicks_dense)

    # early fusion
    concat1 = concatenate([numerics_input, prices_input, clicks_input])
    concat1 = Dense(units=128, activation=dense_act, kernel_initializer=kernel_initializer)(concat1)
    concat1 = BatchNormalization()(concat1)
    concat1 = Dropout(0.2)(concat1)

    # late fusion
    concat2 = concatenate([concat1, numerics_dense, prices_dense, clicks_dense])
    concat2 = Dense(units=128, activation=dense_act, kernel_initializer=kernel_initializer)(concat2)
    concat2 = BatchNormalization()(concat2)
    concat2 = Dropout(0.2)(concat2)

    # last hidden layer
    h = Dense(units=64, activation=dense_act, kernel_initializer=kernel_initializer)(concat2)

    output_layer = Dense(25, activation='softmax')(h)

    model = Model(inputs=[numerics_input, prices_input, clicks_input], outputs=output_layer)

    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
