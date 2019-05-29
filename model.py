from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras import backend as K
from tcn import TCN

from utils import get_logger

logger = get_logger('model')


def compute_receptive_field(tcn_params):
    # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
    receptive_field = tcn_params['nb_stacks']*tcn_params['kernel_size']*(tcn_params['dilations'][-1])
    logger.info(f'Receptive field is: {receptive_field}')
    return receptive_field


def build_model(n_cfs, params, act='relu'):
    K.clear_session()
    imp_tcn_params = params['imp_tcn']
    price_tcn_params = params['price_tcn']
    hist_tcn_params = params['hist_tcn']
    early_fusion_tcn_params = params['early_tcn']

    # impression(meta encoded) input
    imp_input = Input(shape=(None, 157), name='imp_input')
    imp_tcn = TCN(**imp_tcn_params)(imp_input)
    imp_tcn = BatchNormalization()(imp_tcn)

    # price input
    price_input = Input(shape=(None, 2), name='price_input')
    price_tcn = TCN(**price_tcn_params)(price_input)
    price_tcn = BatchNormalization()(price_tcn)

    # history
    hist_input = Input(shape=(None, 4), name='hist_input')
    hist_tcn = TCN(**hist_tcn_params)(hist_input)
    hist_tcn = BatchNormalization()(hist_tcn)

    # c_filter
    c_filter_input = Input(shape=(n_cfs,), name='cfilter_input')
    c_filter_dense = Dense(units=8, activation=act)(c_filter_input)
    c_filter_dense = BatchNormalization()(c_filter_dense)
    c_filter_dense = Dropout(0.2)(c_filter_dense)

    # numeric
    numeric_input = Input(shape=(3,), name='numeric_input')
    numeric_dense = Dense(8, activation=act)(numeric_input)
    numeric_dense = BatchNormalization()(numeric_dense)

    # concat tcn inputs (early fusion)
    early_fusion = concatenate([imp_input, price_input, hist_input])
    early_fusion = BatchNormalization()(early_fusion)
    early_fusion_tcn = TCN(**early_fusion_tcn_params)(early_fusion)

    # late fusion
    late_fusion = concatenate([early_fusion_tcn, imp_tcn, price_tcn, hist_tcn,
                               c_filter_dense, numeric_dense])
    late_fusion = BatchNormalization()(late_fusion)
    late_fusion = Dense(32, activation=act)(late_fusion)

    # output layer
    output_layer = Dense(25, activation='softmax')(late_fusion)

    # impression_batch, histroy_batch, numeric_batch, price_batch, c_filter_batch
    model = Model(inputs=[imp_input, hist_input, numeric_input, price_input, c_filter_input],
                  outputs=output_layer)

    opt = optimizers.Adam(lr=params['learning_rate'])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
