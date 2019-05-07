from keras import optimizers
from keras.layers import concatenate, Dense, Dropout, Embedding, \
                         Input, Flatten, Conv1D, BatchNormalization, MaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from tcn import TCN


def build_model(n_cfs, batch_size, dense_act='relu'):
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
              'dilations': [1, 2, 4, 8],
              # 'activation': 'norm_relu',
              'use_skip_connections': True,
              'dropout_rate': 0.3,
              'return_sequences': False,
              'name': 'tcn'}

    impression_input = Input(shape=(None, 157), name='impression_input')
    params['name'] = 'impression_tcn'
    impression_tcn = TCN(**params)(impression_input)

    # PRICES
    price_input = Input(shape=(None, 1), name='price_input')
    params['name'] = 'price_tcn'
    price_tcn = TCN(**params)(price_input)

    # CURRENT_FILTERS
    cfilter_input = Input(shape=(n_cfs,), name='cfilter_input')
    cfilter_h = Dense(units=30, activation=dense_act)(cfilter_input)
    cfilter_h = Dropout(0.2)(cfilter_h)

    # concatenate
    concat1 = concatenate([numerics_input, impression_tcn, price_tcn, cfilter_h])
    # concat1 = concatenate([numerics_input, cfilter_h])
    # concat1 = BatchNormalization()(concat1)
    concat1 = Dense(units=30, activation=dense_act)(concat1)
    concat1 = Dropout(0.2)(concat1)

    h = Dense(units=30, activation=dense_act)(concat1)

    output_layer = Dense(25, activation='softmax')(h)

    # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
    model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
                  outputs=output_layer)
    # model = Model(inputs=[numerics_input, cfilter_input],
    #               outputs=output_layer)

    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model






# def build_model(n_cfs, batch_size, dense_act='relu'):
#     K.clear_session()
#
#     # build model =====================================================================================
#     # NUMERICS
#     numerics_input = Input(batch_shape=(batch_size, 3, ), name='numerics_input')
#
#     # IMPRESSIONS
#     params = {'nb_filters': 32,
#               'kernel_size': 3,
#               'nb_stacks': 1,
#               'padding': 'causal',
#               'dilations': [1, 2, 4],
#               # 'activation': 'norm_relu',
#               'use_skip_connections': True,
#               'dropout_rate': 0.3,
#               'return_sequences': False,
#               'name': 'tcn'}
#     impression_input = Input(batch_shape=(batch_size, None, 157), name='impression_input')
#     params['name'] = 'impression_tcn'
#     impression_tcn = TCN(**params)(impression_input)
#
#     # PRICES
#     price_input = Input(batch_shape=(batch_size, None, 1), name='price_input')
#     params['name'] = 'price_tcn'
#     price_tcn = TCN(**params)(price_input)
#
#     # CURRENT_FILTERS
#     cfilter_input = Input(batch_shape=(batch_size, n_cfs, ), name='cfilter_input')
#     cfilter_h = Dense(units=30, activation=dense_act)(cfilter_input)
#     cfilter_h = Dropout(0.2)(cfilter_h)
#
#     # concatenate
#     concat1 = concatenate([numerics_input, impression_tcn, price_tcn, cfilter_h])
#     # concat1 = concatenate([numerics_input, cfilter_h])
#     # concat1 = BatchNormalization()(concat1)
#     concat1 = Dense(units=30, activation=dense_act)(concat1)
#     concat1 = Dropout(0.2)(concat1)
#
#     h = Dense(units=30, activation=dense_act)(concat1)
#
#     output_layer = Dense(25, activation='softmax')(h)
#
#     # [numerics_batch, impressions_batch, prices_batch,  cfilters_batch]
#     model = Model(inputs=[numerics_input, impression_input, price_input, cfilter_input],
#                   outputs=output_layer)
#     # model = Model(inputs=[numerics_input, cfilter_input],
#     #               outputs=output_layer)
#
#     opt = optimizers.Adam(lr=0.001)
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#     return model



# def build_model(n_item_ids, n_city, n_country, n_plat, conv1d_filter_size=5):
#     K.clear_session()
#     # build model =====================================================================================
#     # IMPRESSIONS
#     impression_input = Input(shape=(25,), dtype='int32', name='impression_input')
#     impression_embedding = Embedding(n_item_ids, 20, input_length=25)(impression_input)
#     impression_embedding = SpatialDropout1D(0.2)(impression_embedding)
#     impression_conv1 = Conv1D(16, kernel_size=conv1d_filter_size, activation='relu')(impression_embedding)
#     impression_conv1 = BatchNormalization()(impression_conv1)
#     impression_conv1 = MaxPooling1D(2)(impression_conv1)
#     impression_conv1 = Dropout(0.2)(impression_conv1)

#     impression_conv2 = Conv1D(32, kernel_size=conv1d_filter_size, activation='relu')(impression_conv1)
#     impression_conv2 = BatchNormalization()(impression_conv2)
#     impression_conv2 = MaxPooling1D(2)(impression_conv2)
#     impression_conv2 = Dropout(0.2)(impression_conv2)

#     impression_flatten = Flatten()(impression_conv2)
#     impression_flatten = Dropout(0.2)(impression_flatten)

#     # PRICE
#     price_input = Input(shape=(25, 1), name='price_input')
#     price_conv1 = Conv1D(8, kernel_size=conv1d_filter_size, activation='relu')(price_input)
#     price_conv1 = BatchNormalization()(price_conv1)
#     price_conv1 = MaxPooling1D(2)(price_conv1)
#     price_conv2 = Conv1D(16, kernel_size=conv1d_filter_size, activation='relu')(price_conv1)
#     price_conv2 = BatchNormalization()(price_conv2)
#     price_conv2 = MaxPooling1D(2)(price_conv2)
#     price_flatten = Flatten()(price_conv2)

#     # CITY
#     city_input = Input(shape=(1,), dtype='int32', name='city_input')
#     city_embedding = Embedding(n_city, 9, input_length=1)(city_input)
#     city_embedding = SpatialDropout1D(0.1)(city_embedding)
#     city_embedding = Flatten()(city_embedding)

#     # COUNTRY
#     country_input = Input(shape=(1,), dtype='int32', name='country_input')
#     country_embedding = Embedding(n_country, 4, input_length=1)(country_input)
#     country_embedding = SpatialDropout1D(0.1)(country_embedding)
#     country_embedding = Flatten()(country_embedding)

#     # PLATFORM
#     plat_input = Input(shape=(1,), dtype='int32', name='platform')
#     plat_embedding = Embedding(n_plat, 3, input_length=1)(plat_input)
#     plat_embedding = SpatialDropout1D(0.1)(plat_embedding)
#     plat_embedding = Flatten()(plat_embedding)

#     # DEVICE
#     device_input = Input(shape=(2,), name='device_input')

#     # concatenate
#     concat1 = concatenate([impression_flatten, price_flatten])
#     concat1 = BatchNormalization()(concat1)
#     concat1 = Dense(units=30, activation='relu')(concat1)
#     concat1 = Dropout(0.2)(concat1)

#     concat2 = concatenate([concat1, city_embedding, country_embedding, plat_embedding, device_input])
#     concat2 = BatchNormalization()(concat2)
#     concat2 = Dropout(0.2)(concat2)

#     h = Dense(units=30, activation='relu')(concat2)
#     h = Dense(units=30, activation='relu')(h)

#     output_layer = Dense(25, activation='softmax')(h)

#     # [imps, ps, cis, cos, plats, ds], ys
#     model = Model(inputs=[impression_input, price_input, city_input, country_input, plat_input, device_input],
#                   outputs=output_layer)

#     opt = optimizers.Adam(lr=0.001)
#     model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
#     return model