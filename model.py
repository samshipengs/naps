from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras import backend as K
from tcn import TCN

from utils import get_logger

logger = get_logger('model')


def build_model(n_cfs, params, dense_act='relu', kernel_initializer='he_uniform'):
    K.clear_session()
    tcn_params = params['tcn_params']
    # build model =====================================================================================
    # numerics
    numerics_input = Input(shape=(6,), name='numerics_input')
    numerics_dense = Dense(16, activation=dense_act, kernel_initializer=kernel_initializer)(numerics_input)
    numerics_dense = BatchNormalization()(numerics_dense)

    # impressions and prices
    # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
    receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
    logger.info(f'Receptive field is: {receptive_field}')
    merged_input = Input(shape=(None, 159), name='merged_input')
    tcn_params['name'] = 'price_tcn'
    merged_tcn = TCN(**tcn_params)(merged_input)
    merged_tcn_dense = Dense(units=32, activation=dense_act, kernel_initializer=kernel_initializer,
                             name='merged_tcn_dense')(merged_tcn)
    merged_tcn_dense = BatchNormalization()(merged_tcn_dense)

    # CURRENT_FILTERS
    cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
    cfilter_dense1 = Dense(units=16, activation=dense_act, kernel_initializer=kernel_initializer,
                           name='cfilter_dense1')(cfilter_input)
    cfilter_dense1 = BatchNormalization()(cfilter_dense1)
    cfilter_dense2 = Dense(units=8, activation=dense_act, kernel_initializer=kernel_initializer,
                           name='cfilter_dense2')(cfilter_dense1)
    cfilter_dense2 = BatchNormalization()(cfilter_dense2)

    # concatenate
    concat1 = concatenate([numerics_input, merged_tcn, cfilter_dense1])
    concat1 = Dense(units=64, activation=dense_act, kernel_initializer=kernel_initializer)(concat1)
    concat1 = BatchNormalization()(concat1)
    concat1 = Dropout(0.2)(concat1)

    # concatenate
    concat2 = concatenate([concat1, numerics_dense, merged_tcn_dense, cfilter_dense2])
    concat2 = Dense(units=64, activation=dense_act, kernel_initializer=kernel_initializer)(concat2)
    concat2 = BatchNormalization()(concat2)
    concat2 = Dropout(0.2)(concat2)

    # last hidden layer
    h = Dense(units=32, activation=dense_act, kernel_initializer=kernel_initializer)(concat2)

    output_layer = Dense(25, activation='softmax')(h)

    model = Model(inputs=[numerics_input, merged_input, cfilter_input], outputs=output_layer)

    opt = optimizers.Adam(lr=params['learning_rate'])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


# def build_model(n_cfs, params, dense_act='relu'):
#     K.clear_session()
#     tcn_params = params['tcn_params']
#     # build model =====================================================================================
#     # numerics
#     numerics_input = Input(shape=(6,), name='numerics_input')
#     numerics_dense = Dense(16, activation=dense_act)(numerics_input)
#
#     # impressions and prices
#     # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
#     receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
#     logger.info(f'Receptive field is: {receptive_field}')
#     merged_input = Input(shape=(None, 159), name='merged_input')
#     tcn_params['name'] = 'price_tcn'
#     merged_tcn = TCN(**tcn_params)(merged_input)
#     # merged_tcn_dense = Dense(units=32, activation=dense_act, name='merged_tcn_dense')(merged_tcn)
#
#     # CURRENT_FILTERS
#     cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
#     cfilter_dense1 = Dense(units=16, activation=dense_act, name='cfilter_dense1')(cfilter_input)
#     # cfilter_dense2 = Dropout(0.2)(cfilter_dense1)
#
#     # concatenate
#     concat1 = concatenate([numerics_input, merged_tcn, cfilter_dense1])
#     concat1 = Dense(units=64, activation=dense_act)(concat1)
#     concat1 = Dropout(0.2)(concat1)
#
#     # # concatenate
#     # concat2 = concatenate([concat1, numerics_dense, price_dense,
#     #                        cfilter_dense2])
#     # concat2 = Dense(units=64, activation=dense_act)(concat2)
#     # concat2 = Dropout(0.2)(concat2)
#
#     # last hidden layer
#     h = Dense(units=32, activation=dense_act)(concat1)
#
#     output_layer = Dense(25, activation='softmax')(h)
#
#     model = Model(inputs=[numerics_input, merged_input, cfilter_input], outputs=output_layer)
#
#     opt = optimizers.Adam(lr=params['learning_rate'])
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#     return model
