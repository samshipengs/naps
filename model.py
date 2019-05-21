from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras import backend as K
from tcn import TCN

from utils import get_logger

logger = get_logger('model')


def build_model(n_cfs, params, dense_act='relu'):
    K.clear_session()
    tcn_params = params['tcn_params']
    # build model =====================================================================================
    # NUMERICS
    numerics_input = Input(shape=(6,), name='numerics_input')
    numerics_dense = Dense(16, activation=dense_act)(numerics_input)

    # IMPRESSIONS
    # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
    receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
    logger.info(f'Receptive field is: {receptive_field}')


    # CURRENT_FILTERS
    cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
    cfilter_dense1 = Dense(units=16, activation=dense_act)(cfilter_input)
    cfilter_dense2 = Dropout(0.2)(cfilter_dense1)

    # PRICES
    price_input = Input(shape=(None, 1), name='price_input')
    tcn_params['name'] = 'price_tcn'
    price_tcn = TCN(**tcn_params)(price_input)
    price_dense = Dense(units=32, activation=dense_act)(price_tcn)

    # concatenate
    concat1 = concatenate([numerics_input, price_tcn, cfilter_dense1])
    concat1 = Dense(units=64, activation=dense_act)(concat1)
    concat1 = Dropout(0.2)(concat1)

    # concatenate
    concat2 = concatenate([concat1, numerics_dense, price_dense,
                           cfilter_dense2])
    concat2 = Dense(units=64, activation=dense_act)(concat2)
    concat2 = Dropout(0.2)(concat2)

    # last hidden layer
    h = Dense(units=32, activation=dense_act)(concat2)

    output_layer = Dense(25, activation='softmax')(h)

    # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
    model = Model(inputs=[numerics_input, price_input, cfilter_input],
                  outputs=output_layer)

    opt = optimizers.Adam(lr=params['learning_rate'])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


# def build_model(n_cfs, params, dense_act='relu'):
#     K.clear_session()
#     tcn_params = params['tcn_params']
#     # build model =====================================================================================
#     # NUMERICS
#     numerics_input = Input(shape=(6,), name='numerics_input')
#     numerics_dense = Dense(16, activation=dense_act)(numerics_input)
#
#     # IMPRESSIONS
#     # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
#     receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
#     logger.info(f'Receptive field is: {receptive_field}')
#
#     impression_input = Input(shape=(None, 157), name='impression_input')
#     tcn_params['name'] = 'impression_tcn'
#     impression_tcn = TCN(**tcn_params)(impression_input)
#     impression_dense = Dense(32, activation=dense_act)(impression_tcn)
#
#     # CURRENT_FILTERS
#     cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
#     cfilter_dense1 = Dense(units=16, activation=dense_act)(cfilter_input)
#     cfilter_dense2 = Dropout(0.2)(cfilter_dense1)
#
#     # PRICES
#     price_input = Input(shape=(None, 1), name='price_input')
#     tcn_params['name'] = 'price_tcn'
#     price_tcn = TCN(**tcn_params)(price_input)
#     price_dense = Dense(units=32, activation=dense_act)(price_tcn)
#
#     # concatenate
#     concat1 = concatenate([numerics_input, price_tcn, impression_tcn, cfilter_dense1])
#     concat1 = Dense(units=64, activation=dense_act)(concat1)
#     concat1 = Dropout(0.2)(concat1)
#
#     # concatenate
#     concat2 = concatenate([concat1, numerics_dense, price_dense, impression_dense,
#                            cfilter_dense2])
#     concat2 = Dense(units=64, activation=dense_act)(concat2)
#     concat2 = Dropout(0.2)(concat2)
#
#     # last hidden layer
#     h = Dense(units=32, activation=dense_act)(concat2)
#
#     output_layer = Dense(25, activation='softmax')(h)
#
#     # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
#     model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
#                   outputs=output_layer)
#
#     opt = optimizers.Adam(lr=params['learning_rate'])
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#     return model


# def build_model(n_cfs, params, dense_act='relu'):
#     K.clear_session()
#     tcn_params = params['tcn_params']
#     # build model =====================================================================================
#     # NUMERICS
#     numerics_input = Input(shape=(6,), name='numerics_input')
#     numerics = Dense(8, activation=dense_act)(numerics_input)
#
#     # IMPRESSIONS
#     # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
#     receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
#     logger.info(f'Receptive field is: {receptive_field}')
#
#     impression_input = Input(shape=(None, 157), name='impression_input')
#     tcn_params['name'] = 'impression_tcn'
#     impression_tcn = TCN(**tcn_params)(impression_input)
#     # CURRENT_FILTERS
#     cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
#     cfilter_h = Dense(units=32, activation=dense_act)(cfilter_input)
#     cfilter_h = Dropout(0.2)(cfilter_h)
#     impression_cfilter_concat = concatenate([impression_tcn, cfilter_h])
#     impression_cfilter_h = Dense(units=32, activation=dense_act)(impression_cfilter_concat)
#
#     # PRICES
#     price_input = Input(shape=(None, 1), name='price_input')
#     tcn_params['name'] = 'price_tcn'
#     price_tcn = TCN(**tcn_params)(price_input)
#     # price_tcn = BatchNormalization()(price_tcn)
#
#     # concatenate
#     concat = concatenate([numerics, price_tcn, impression_cfilter_h])
#     # concat1 = BatchNormalization()(concat1)
#     concat = Dense(units=128, activation=dense_act)(concat)
#     concat = Dropout(0.2)(concat)
#
#     # last hidden layer
#     h = Dense(units=64, activation=dense_act)(concat)
#
#     output_layer = Dense(25, activation='softmax')(h)
#
#     # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
#     model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
#                   outputs=output_layer)
#
#     opt = optimizers.Adam(lr=params['learning_rate'])
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#     return model



# def build_model(n_cfs, params, dense_act='relu'):
#     K.clear_session()
#     tcn_params = params['tcn_params']
#     # build model =====================================================================================
#     # NUMERICS
#     numerics_input = Input(shape=(4,), name='numerics_input')
#     numerics = Dense(16, activation=dense_act)(numerics_input)

#     # IMPRESSIONS
#     # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
#     receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
#     logger.info(f'Receptive field is: {receptive_field}')

#     impression_input = Input(shape=(None, 157), name='impression_input')
#     tcn_params['name'] = 'impression_tcn'
#     impression_tcn = TCN(**tcn_params)(impression_input)
#     # impression_tcn = BatchNormalization()(impression_tcn)

#     # PRICES
#     price_input = Input(shape=(None, 1), name='price_input')
#     tcn_params['name'] = 'price_tcn'
#     price_tcn = TCN(**tcn_params)(price_input)
#     # price_tcn = BatchNormalization()(price_tcn)

#     # CURRENT_FILTERS
#     cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
#     cfilter_h = Dense(units=32, activation=dense_act)(cfilter_input)
#     # cfilter_h = BatchNormalization()(cfilter_h)
#     cfilter_h = Dropout(0.2)(cfilter_h)

#     # concatenate
#     concat1 = concatenate([impression_tcn, price_tcn, cfilter_h])
#     # concat1 = BatchNormalization()(concat1)
#     concat1 = Dense(units=64, activation=dense_act)(concat1)
#     concat1 = Dropout(0.2)(concat1)

#     concat2 = concatenate([numerics, concat1])
#     concat2 = Dense(units=128, activation=dense_act)(concat2)
#     concat2 = Dropout(0.2)(concat2)

#     h = Dense(units=64, activation=dense_act)(concat2)

#     output_layer = Dense(25, activation='softmax')(h)

#     # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
#     model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
#                   outputs=output_layer)

#     opt = optimizers.Adam(lr=params['learning_rate'])
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#     return model
