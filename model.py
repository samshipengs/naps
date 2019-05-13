# from keras import optimizers
# from keras.layers import concatenate, Dense, Dropout, Input, BatchNormalization
# from keras.models import Model
# from keras import backend as K
# from tcn import TCN
#
# from utils import get_logger
#
# logger = get_logger('model')
#
#
# def build_model(n_cfs, params, dense_act='relu'):
#     K.clear_session()
#     tcn_params = params['tcn_params']
#     # build model =====================================================================================
#     # NUMERICS
#     numerics_input = Input(shape=(3,), name='numerics_input')
#
#     # IMPRESSIONS
#     # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
#     receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
#     logger.info(f'Receptive field is: {receptive_field}')
#
#     impression_input = Input(shape=(None, 157), name='impression_input')
#     tcn_params['name'] = 'impression_tcn'
#     impression_tcn = TCN(**tcn_params)(impression_input)
#     # impression_tcn = BatchNormalization()(impression_tcn)
#
#     # PRICES
#     price_input = Input(shape=(None, 1), name='price_input')
#     tcn_params['name'] = 'price_tcn'
#     price_tcn = TCN(**tcn_params)(price_input)
#     # price_tcn = BatchNormalization()(price_tcn)
#
#     # CURRENT_FILTERS
#     cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
#     cfilter_h = Dense(units=32, activation=dense_act)(cfilter_input)
#     # cfilter_h = BatchNormalization()(cfilter_h)
#     cfilter_h = Dropout(0.2)(cfilter_h)
#
#     # concatenate
#     concat1 = concatenate([numerics_input, impression_tcn, price_tcn, cfilter_h])
#     # concat1 = BatchNormalization()(concat1)
#     concat1 = Dense(units=64, activation=dense_act)(concat1)
#     concat1 = Dropout(0.2)(concat1)
#
#     h = Dense(units=32, activation=dense_act)(concat1)
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
    numerics_input = Input(shape=(3,), name='numerics_input')
    numerics = Dense(16, activation=dense_act)(numerics_input)

    # IMPRESSIONS
    # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
    receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
    logger.info(f'Receptive field is: {receptive_field}')

    impression_input = Input(shape=(None, 157), name='impression_input')
    tcn_params['name'] = 'impression_tcn'
    impression_tcn = TCN(**tcn_params)(impression_input)
    # impression_tcn = BatchNormalization()(impression_tcn)

    # PRICES
    price_input = Input(shape=(None, 1), name='price_input')
    tcn_params['name'] = 'price_tcn'
    price_tcn = TCN(**tcn_params)(price_input)
    # price_tcn = BatchNormalization()(price_tcn)

    # CURRENT_FILTERS
    cfilter_input = Input(shape=(n_cfs, ), name='cfilter_input')
    cfilter_h = Dense(units=32, activation=dense_act)(cfilter_input)
    # cfilter_h = BatchNormalization()(cfilter_h)
    cfilter_h = Dropout(0.2)(cfilter_h)

    # concatenate
    concat1 = concatenate([impression_tcn, price_tcn, cfilter_h])
    # concat1 = BatchNormalization()(concat1)
    concat1 = Dense(units=64, activation=dense_act)(concat1)
    concat1 = Dropout(0.2)(concat1)

    concat2 = concatenate([numerics, concat1])
    concat2 = Dense(units=128, activation=dense_act)(concat2)
    concat2 = Dropout(0.2)(concat2)

    h = Dense(units=64, activation=dense_act)(concat2)

    output_layer = Dense(25, activation='softmax')(h)

    # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
    model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
                  outputs=output_layer)

    opt = optimizers.Adam(lr=params['learning_rate'])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
