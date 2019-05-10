from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, \
                         Input, Flatten, Conv1D, BatchNormalization, MaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from tcn import TCN


def build_model(n_cfs, dense_act='relu'):
    K.clear_session()

    # build model =====================================================================================
    # NUMERICS
    numerics_input = Input(shape=(3,), name='numerics_input')

    # IMPRESSIONS
    # Receptive field = nb_stacks_of_residuals_blocks * kernel_size * last_dilation.
    params = {'nb_filters': 32,
              'kernel_size': 3,
              'nb_stacks': 2,
              'padding': 'causal',
              'dilations': [1, 2, 4],
              # 'activation': 'norm_relu',
              'use_skip_connections': True,
              'dropout_rate': 0.2,
              'return_sequences': False,
              'name': 'tcn'}

    impression_input = Input(shape=(None, 157), name='impression_input')
    params['name'] = 'impression_tcn'
    impression_tcn = TCN(**params)(impression_input)
    impression_tcn = BatchNormalization()(impression_tcn)

    # PRICES
    price_input = Input(shape=(None, 1), name='price_input')
    params['name'] = 'price_tcn'
    price_tcn = TCN(**params)(price_input)
    price_tcn = BatchNormalization()(price_tcn)

    # CURRENT_FILTERS
    cfilter_input = Input(shape=(n_cfs,), name='cfilter_input')
    cfilter_h = Dense(units=32, activation=dense_act)(cfilter_input)
    cfilter_h = BatchNormalization()(cfilter_h)
    cfilter_h = Dropout(0.2)(cfilter_h)

    # concatenate
    concat1 = concatenate([numerics_input, impression_tcn, price_tcn, cfilter_h])
    # concat1 = BatchNormalization()(concat1)
    concat1 = Dense(units=64, activation=dense_act)(concat1)
    concat1 = Dropout(0.2)(concat1)

    h = Dense(units=32, activation=dense_act)(concat1)

    output_layer = Dense(25, activation='softmax')(h)

    # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
    model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
                  outputs=output_layer)
    # model = Model(inputs=[numerics_input, cfilter_input],
    #               outputs=output_layer)

    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
