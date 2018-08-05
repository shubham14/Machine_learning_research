
# import general packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from os import path
from sklearn.cluster import KMeans
import h5py

# import keras packages
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Activation, GlobalAveragePooling2D, merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers import merge
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.utils import plot_model
from keras.optimizers import *
from keras import backend as K
import argparse
K.set_image_data_format('channels_last')

# rb_mast orientation for 'unrotation'
def orient_rb(data_in, rb):
    n_dep, n_az = data_in.shape[0:2]
    assert (n_dep == len(rb))

    data_out = np.zeros_like(data_in)
    deg_az = 360. / n_az

    for id in range(n_dep):
        idx_rotated = np.mod(int(np.round(rb[id] / deg_az)) + np.arange(n_az), n_az)
        data_out[id] = data_in[id, idx_rotated]

    return data_out


def removeDuplicates(dep, data):
    _, idxUnq = np.unique(dep, return_index=True)  # sorted array
    dep = dep[idxUnq]
    data = data[idxUnq]
    return dep, data


def sampling(args):
    # z_log_var is taken as 0
    z_mean = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.) * epsilon


# interpolate azimuthal data
def interpAz(data, multiple):
    nDep, nAz = data.shape
    xp = np.arange(0, 360, 360/nAz)
    x = np.arange(0, 360, 360/nAz/multiple)
    data_out = np.array([np.interp(x, xp, row, period=360) for row in data])
    return data_out


# plot loss function
def plot_loss(losses, save=False, saveFileName=None):
    plt.figure(figsize=(10, 10))
    g_loss = np.array(losses['g'])
    d_loss = np.array(losses['d'])
    grad_penalty = np.array(losses['grad'])
    info_penalty = np.array(losses['info'])
    plt.plot(g_loss, label='G loss')
    plt.plot(d_loss, label='D loss')
    plt.plot(grad_penalty, label='grad penalty')
    plt.plot(info_penalty, label='info penalty')
    plt.legend()
    if save:
        if saveFileName is not None:
            plt.savefig(saveFileName)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


def get_noise(dim_noise, dim_cat, dim_cont, batch_size=32):
    noise = np.random.normal(0, 1, size=(batch_size, dim_noise))
    label = np.random.randint(0, dim_cat, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=dim_cat)
    cont_2d = np.random.uniform(-1, 1, size=(batch_size, dim_cont))
    return noise, label, cont_2d


# plot generated images
def plot_gen(generator, dim, figsize=(10, 10), channel=0, save=False, saveFileName=None, method='cat', **kwargs):
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    dim_cont = generator.layers[2].input_shape[1]
    plt.figure(figsize=figsize)
    n_image_row, n_image_col = dim
    if method is 'cat':
        for i in range(dim_cat):
            noise, _, _ = get_noise(dim_noise, dim_cat, dim_cont, batch_size=n_image_col)
            label = np.repeat(i, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=dim_cat)
            cont = np.repeat(np.zeros((1, dim_cont)), n_image_col, axis=0)
            image_gen = generator.predict([noise, label, cont])
            for j in range(n_image_col):
                plt.subplot(dim_cat, n_image_col, i*n_image_col+j+1)
                plt.imshow(image_gen[j, :, :, channel],  **kwargs)
                plt.axis('off')
    elif method is 'cont':
        label_number = 0
        cont_range_row = np.linspace(-1, 1, num=n_image_row)
        cont_range_col = np.linspace(-1, 1, num=n_image_col)
        for i in range(n_image_row):
            noise, _, _ = get_noise(dim_noise, dim_cat, dim_cont, batch_size=n_image_col)
            label = np.repeat(label_number, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=dim_cat)
            cont = np.concatenate([np.array([cont_range_row[i], cont_range_col[j]]).reshape(1, -1) for j in range(n_image_col)])
            image_gen = generator.predict([noise, label, cont])
            for j in range(n_image_col):
                plt.subplot(n_image_row, n_image_col, i*n_image_col+j+1)
                plt.imshow(image_gen[j, :, :, channel], cmap='gray')
                plt.axis('off')
    else:
        raise ValueError('Wrong method!')
    if save:
        if saveFileName is not None:
            plt.savefig(saveFileName)
    else:
        plt.tight_layout()
        plt.show()
    plt.clf()
    plt.close('all')


def build_generator_mnist_test(n_class, n_cont, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, leaky_relu_alpha=0.2):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(n_class,))
    g_in_cont = Input(shape=(n_cont,))
    g_in = concatenate([g_in_noise, g_in_cat, g_in_cont])
    x = Dense((n_rows//4) * (n_cols//4) * n_first_conv_ch)(g_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Reshape((n_rows // 4, n_cols // 4, n_first_conv_ch))(x)
    x = Conv2DTranspose(n_first_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2DTranspose(n_first_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    g_out = Conv2DTranspose(n_out_ch, (4, 4), strides=1, padding='same', activation='sigmoid')(x)
    generator = Model([g_in_noise, g_in_cat, g_in_cont], g_out, name='generator')
    print('Summary of Generator (for InfoWGAN-GP)')
    generator.summary()
    return generator


def build_generator(n_class, n_cont, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=128, dim_noise=100, n_feat=9):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(n_class,))
    g_in_cont = Input(shape=(n_feat,))
    g_in = concatenate([g_in_noise, g_in_cat, g_in_cont])
    x = Dense((n_rows//8) * (n_cols//8) * n_first_conv_ch)(g_in)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Reshape((n_rows // 8, n_cols // 8, n_first_conv_ch))(x)
    x = Conv2DTranspose(n_first_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(n_first_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(n_first_conv_ch//8, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(n_out_ch, (4, 4), strides=1, padding='same', activation='sigmoid')(x)
    generator = Model([g_in_noise, g_in_cat, g_in_cont], g_out, name='generator')
    print('Summary of Generator (for InfoWGAN-GP)')
    generator.summary()
    return generator


def build_disc_aux_mnist_test(n_class, n_cont, n_rows, n_cols, n_in_ch=1, n_last_conv_ch=128, leaky_relu_alpha=0.2):
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(n_last_conv_ch//4, (4, 4), strides=1, padding='same')(d_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha, name='d_feature')(x)
    d_out_base = Flatten()(x)

    # discriminator value output (for Wasserstein loss)
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    d_out_val = Dense(1)(x)  # no activation function

    # categorical output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    c_out_softmax = Dense(n_class, activation='softmax')(x)

    # continuous output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    f_out_mean = Dense(n_cont)(x)
    f_out_logstd = Dense(n_cont)(x)
    f_out_gaussian = Lambda(sampling, output_shape=(n_cont, ), name='cont')([z_mean, z_log_var])
    # similar to averaging layer 
    f_out_gaussian = Dense(1)(f_out_gaussian)

    # discriminator model
    discriminator = Model(d_in, d_out_val, name='discriminator')
    print('Summary of Discriminator (for InfoWGAN-GP)')
    discriminator.summary()

    # classifier model
    classifier = Model(d_in, c_out_softmax, name='classifier')
    print('Summary of Classifier (for InfoWGAN-GP)')
    classifier.summary()

    # feature extractor model
    feature_extractor = Model(d_in, f_out_gaussian, name='feature_extractor')
    print('Summary of Feature Extractor (for InfoWGAN-GP)')
    feature_extractor.summary()

    return discriminator, classifier, feature_extractor


def build_disc_aux(n_class, n_cont, n_rows, n_cols, n_in_ch=1, n_last_conv_ch=128, leaky_relu_alpha=0.2):
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(n_last_conv_ch//8, (4, 4), strides=1, padding='same')(d_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=leaky_relu_alpha, name='d_feature')(x)
    d_out_base = Flatten()(x)

    # discriminator value output (for Wasserstein loss)
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    d_out_val = Dense(1)(x)  # no activation function

    # categorical output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    c_out_softmax = Dense(n_class, activation='softmax')(x)

    # continuous output
    x = Dense(1024)(d_out_base)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    f_out_mean = Dense(n_cont)(x)
    # why approximating the log-std
    f_out_logstd = Dense(n_cont)(x)
    f_out_gaussian = Lambda(sampling, output_shape=(n_cont, ), name='cont')([f_out_mean, f_out_logstd])
    # similar to averaging layer 
    f_out_gaussian = Dense(1)(f_out_gaussian)

    # discriminator model
    discriminator = Model(d_in, d_out_val, name='discriminator')
    print('Summary of Discriminator (for InfoWGAN-GP)')
    discriminator.summary()

    # classifier model
    classifier = Model(d_in, c_out_softmax, name='classifier')
    print('Summary of Classifier (for InfoWGAN-GP)')
    classifier.summary()

    # feature extractor model
    feature_extractor = Model(d_in, [f_out_gaussian, f_out_mean], name='feature_extractor')
    print('Summary of Feature Extractor (for InfoWGAN-GP)')
    feature_extractor.summary()

    return discriminator, classifier, feature_extractor


def train_infowgangp(image_set, label_set, generator, generator_opt, discriminator, discriminator_opt, classifier, feature_extractor, losses, lambda_gp=10,
                     lambda_info=1, batch_size=32, n_epochs=100, train_dgratio=5, save_every=10, save_filename_prefix=None):
    real_in = Input(shape=discriminator.layers[0].input_shape[1:])
    label_in = Input(shape=generator.layers[1].input_shape[1:])
    gen_in_noise = Input(shape=generator.layers[0].input_shape[1:])
    gen_in_cat = Input(shape=generator.layers[1].input_shape[1:])
    gen_in_cont = Input(shape=generator.layers[2].input_shape[1:])
    gen_out = generator([gen_in_noise, gen_in_cat, gen_in_cont])
    disc_out_val_real = discriminator(real_in)
    disc_out_val_fake = discriminator(gen_out)
    class_out_softmax_real = classifier(real_in)
    class_out_softmax_fake = classifier(gen_out)
    [feature_out_gaussian, feature_out_mean] = feature_extractor(gen_out)
    # feature_out_mean = feature_out_gaussian[:, :, 0]
    # feature_out_logstd = feature_out_gaussian[:, :, 1]

    # gradient penalty
    eps_in = K.placeholder(shape=(None, 1, 1, 1))
    interp = eps_in * real_in + (1 - eps_in) * gen_out
    grad_interp = discriminator(interp)
    grad = K.gradients(grad_interp, [interp])[0]
    grad_norm = K.sqrt(K.sum(K.square(grad), axis=np.arange(1, len(grad.shape))))
    grad_penalty = K.mean(K.square(grad_norm - 1))

    # information maximization penalty
    info_penalty_cat_real = K.mean(-K.sum(K.log(class_out_softmax_real + K.epsilon()) * label_in, axis=1))
    info_penalty_cat_fake = K.mean(-K.sum(K.log(class_out_softmax_fake + K.epsilon()) * gen_in_cat, axis=1))
    info_penalty_cat = info_penalty_cat_real + info_penalty_cat_fake
    # feature_out_logstd is taken as 0
    cont_norm = (gen_in_cont - feature_out_mean) / (K.exp(0) + K.epsilon())
    info_penalty_cont = K.mean(K.sum(0. + 0.5 * K.square(cont_norm), axis=1))
    info_penalty = info_penalty_cat + info_penalty_cont

    # loss
    d_loss_real = K.mean(disc_out_val_real)
    d_loss_fake = K.mean(disc_out_val_fake)

    g_loss = -d_loss_fake + lambda_info * (info_penalty_cat_fake + info_penalty_cont)
    g_train_updates = generator_opt.get_updates(generator.trainable_weights, [], g_loss)
    g_train = K.function([gen_in_noise, gen_in_cat, gen_in_cont],
                         [g_loss],
                         g_train_updates)

    d_loss = d_loss_fake - d_loss_real + lambda_gp * grad_penalty + lambda_info * info_penalty
    d_train_updates = discriminator_opt.get_updates(discriminator.trainable_weights, [], d_loss)
    d_train = K.function([real_in, label_in, gen_in_noise, gen_in_cat, gen_in_cont, eps_in],
                         [d_loss_real, d_loss_fake, grad_penalty, info_penalty],
                         d_train_updates)

    # training
    n_train = image_set.shape[0]
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    dim_cont = generator.layers[2].input_shape[1]
    n_ch = discriminator.layers[0].input_shape[-1]
    for ie in range(n_epochs):
        print('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = n_train // batch_size
        progbar = generic_utils.Progbar(n_batches*batch_size)
        for ib in range(n_batches):
            # real batch
            idx_batch = idx_randperm[range(ib * batch_size, ib * batch_size + batch_size)]
            image_real_batch = image_set[idx_batch]
            label_real_batch = label_set[idx_batch]
            label_real_batch = np_utils.to_categorical(label_real_batch, num_classes=dim_cat)
            # fake batch
            noise_disc, label_disc, cont_disc = get_noise(dim_noise, dim_cat, dim_cont, batch_size=batch_size)
            # train the discriminator model
            eps = np.random.uniform(size=(batch_size, 1, 1, 1))
            d_loss_real_train_val, d_loss_fake_train_val, grad_pen_train_val, info_pen_train_val = d_train([image_real_batch, label_real_batch, noise_disc, label_disc, cont_disc, eps])
            d_loss_train_val = d_loss_fake_train_val - d_loss_real_train_val + lambda_gp * grad_pen_train_val + lambda_info * info_pen_train_val
            losses['d'].append(d_loss_train_val)
            losses['grad'].append(grad_pen_train_val)
            losses['info'].append(info_pen_train_val)
            # train generator and classifier
            if (ib + 1) % train_dgratio == 0:
                noise_gen, label_gen, cont_gen = get_noise(dim_noise, dim_cat, dim_cont, batch_size=batch_size)
                g_loss_train_val, = g_train([noise_gen, label_gen, cont_gen])
                losses['g'].extend([g_loss_train_val] * train_dgratio)
                # update progress bar
                progbar.add(batch_size * train_dgratio, values=[('G loss', g_loss_train_val),
                                                                ('D loss', d_loss_train_val),
                                                                ('grad penalty', grad_pen_train_val),
                                                                ('info penalty', info_pen_train_val)])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            # display generated images channel by channel
            for ic in range(n_ch):
                save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                plot_gen(generator, (dim_cat, dim_cat), (15, 15), ic, True, save_filename_image_gen, method='cat')
                save_filename_image_gen = '%s_cont_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                plot_gen(generator, (dim_cat, dim_cat), (15, 15), ic, True, save_filename_image_gen, method='cont')

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)


if __name__ == '__main__':
    losses = {'g': [], 'd': [], 'grad': [], 'info': []}
    n_row, n_col = 28, 28
    nc = 10
    nz = 8
    batch_size = 128
    n_epochs = 500
    n_save_every = 10

    # hyperparameters
    g_lr = 2e-4
    g_beta1 = 0.5
    g_beta2 = 0.9
    d_lr = 2e-4
    n_strides2 = 3
    d_beta1 = 0.5
    d_beta2 = 0.9
    dim_noise = 10
    n_class = 5  # changes according to the number of clusters you want to obtain
    n_cont = 2

    workMode = 'FFTCWT'

    # data folder path
    dataPath = ['/home/SDash2/Desktop/Gullfaks/',
                '/home/SDash2/Desktop/Gullfaks/',
                '/home/SDash2/Desktop/Gullfaks/']
    dataFilename = ['Gullfaks_1605_2219_%s' % workMode,
                    'Gullfaks_662_1646_%s' % workMode,
                    'Gullfaks_215_700_%s' % workMode]

    modelPath = '/home/SDash2/Desktop/Data/Gullfaks/7in/Results/'
    resultPath = modelPath

    # for continuous label semi-supervision
    stdoffPath = ['/home/SDash2/Desktop/Data/Gullfaks/7in/2219-1624m/A2S/',
                  '/home/SDash2/Desktop/Data/Gullfaks/7in/1644-681m/A2S/',
                  '/home/SDash2/Desktop/Data/Gullfaks/7in/699-26m/A2S/']

    stdOffname = ['Gullfaks_2219_1624_A2S_rb',
                  'Gullfaks_1644_681_A2S_rb',
                  'Gullfaks_699_26_A2S_rb']

    # FFTCWT & STC parameters
    f = np.array([10, 14])  # kHz

    # output folder
    modelPath = '/home/SDash2/Desktop/Data/Gullfaks/7in/Results/'
    save_filename_prefix = '%s_infowgangp_noise%d_cat%d' % (modelPath, dim_noise, n_class)

    depInput = []
    rbmastInput = []
    dataInput = []

    for ifile in range(len(dataPath)):
        dataFullname = dataPath[ifile] + dataFilename[ifile] + '.mat'
        print('Loading sonic data file: %s' % dataFullname)
        hf = h5py.File(dataFullname, 'r')
        depData = np.squeeze(hf['dep'].value)
        depInput.append(depData)
        rbmastData = np.squeeze(hf['rb_sonic'].value)
        rbmastInput.append(rbmastData)
        if workMode.lower() == 'FFTCWT'.lower():
            fSet = np.squeeze(hf['f'].value)
            stcData = hf['WLT_STP'].value.astype('float32')
        elif workMode.lower() == 'STCBPF'.lower():
            fSet = np.squeeze(hf['fBandCtr'].value)
            stcData = hf['stc_all'].value.astype('float32')
        else:
            sys.exit('Invalid work mode!')
        fIdx = np.squeeze(np.where((fSet >= f[0]) & (fSet <= f[1])))
        stcData = stcData[:, :, fIdx, :, :]
        if fIdx.ndim is 0:
            stcData = stcData[:, :, np.newaxis, :, :]
        stcData = np.transpose(stcData, [0, 1, 3, 4, 2])
        dataInput.append(stcData)
        hf.close()
    depInput = np.concatenate(depInput)
    rbmastInput = np.concatenate(rbmastInput)
    dataInput = np.concatenate(dataInput)

    # remove duplicate sonic data
    _, dataInput = removeDuplicates(depInput, dataInput)
    depInput, rbmastInput = removeDuplicates(depInput, rbmastInput)
    Ndepth, Naz, Ntime, Nslow, Nfreq = dataInput.shape
    print 'Before'
    print Ndepth, Naz, Ntime, Nslow, Nfreq

    # crop data for filter size compatibility
    if Ndepth % batch_size:
        Ndepth = (Ndepth / batch_size) * batch_size
        depInput = depInput[0:Ndepth]
        rbmastInput = rbmastInput[0:Ndepth]
        dataInput = dataInput[0:Ndepth, :, :, :, :]
    if Ntime % (2 ** n_strides2):
        Ntime = int(Ntime / (2 ** n_strides2)) * (2 ** n_strides2)
        dataInput = dataInput[:, :, 0:Ntime, :, :]
    if Nslow % (2 ** n_strides2):
        Nslow = int(Nslow / (2 ** n_strides2)) * (2 ** n_strides2)
        dataInput = dataInput[:, :, :, 0:Nslow, :]

    Ndepth_new, Naz_new, Ntime_new, Nslow_new, Nfreq_new = dataInput.shape
    print 'After'
    print Ndepth_new, Naz_new, Ntime_new, Nslow_new, Nfreq_new
    print Ndepth, Naz, Ntime, Nslow, Nfreq

    std_label = []
    for ifile in range(len(stdoffPath)):
        stdFullname = stdoffPath[ifile] + stdOffname[ifile] + '.mat'
        print('Loading sonic data file: %s' % stdFullname)
        hf = h5py.File(stdFullname, 'r')
        ans = hf['a2st_stdoff'].value
        std_label.append(ans)
        hf.close()
    std_label = np.concatenate(std_label)

    _, std_label = removeDuplicates(depInput, std_label)
    std_label = interpAz(std_label, 2)
    a, b = std_label.shape

    dataSet = np.reshape(dataInput, (Ndepth_new * Naz_new, Ntime_new, Nslow_new, Nfreq_new))
    std_label = np.reshape(std_label, (a*Naz, b/Naz))

    # mean of the 9-dimensional features of the continous labels
    std_label = np.mean(std_label, axis=1)
    
    generator = build_generator(n_class, n_cont, Ntime, Nslow, n_out_ch=Nfreq, n_first_conv_ch=256,
                                           dim_noise=dim_noise)
    generator_opt = Adam(g_lr, g_beta1, g_beta2)
    plot_model(generator, to_file='%s_generator.pdf' % save_filename_prefix, show_shapes=True)

    discriminator, classifier, feature_extractor = build_disc_aux(n_class, n_cont, Ntime, Nslow, n_in_ch=3, n_last_conv_ch=256)
    discriminator_opt = Adam(d_lr, d_beta1, d_beta2)
    plot_model(discriminator, to_file='%s_discriminator.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(classifier, to_file='%s_classifier.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(feature_extractor, to_file='%s_feature_extractor.pdf' % save_filename_prefix, show_shapes=True)

    generator_weights_file = '%s_generator_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    discriminator_weights_file = '%s_discriminator_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    classifier_weights_file = '%s_classifier_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    feature_extractor_weights_file = '%s_feature_extractor_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    if path.isfile(generator_weights_file) \
            and path.isfile(discriminator_weights_file)\
            and path.isfile(classifier_weights_file)\
            and path.isfile(feature_extractor_weights_file):
        # load InfoWGAN-GP weights that already existed
        print('loading InfoWGAN-GP model weights')
        generator.load_weights(generator_weights_file)
        discriminator.load_weights(discriminator_weights_file)
        classifier.load_weights(classifier_weights_file)
        feature_extractor.load_weights(feature_extractor_weights_file)
    else:
        # train InfoWGAN-GP
        print('training InfoWGAN-GP model')

        train_infowgangp(dataSet, std_label, generator, generator_opt, discriminator, discriminator_opt, classifier, feature_extractor, losses,
                         lambda_gp=10,
                         lambda_info=1,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         train_dgratio=5,
                         save_every=n_save_every,
                         save_filename_prefix=save_filename_prefix)
        generator.save_weights(generator_weights_file)
        discriminator.save_weights(discriminator_weights_file)
        classifier.save_weights(classifier_weights_file)
        feature_extractor.save_weights(feature_extractor_weights_file)

    def cluster_acc(y_true, y_pred):
        from sklearn.utils.linear_assignment_ import linear_assignment
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w

    # display generated images channel by channel
    for ic in range(Nfreq):
        save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
        plot_gen(generator, (10, 10), (15, 15), ic, True, save_filename_image_gen)

    batch_size_dataset = 31744
    # clustering with discriminator

    print('InfoGAN clustering on all azimuths (%d clusters)' % n_class)
    dataSet_real_prob = discriminator.predict(dataSet)
    dataSet_real_prob = np.reshape(dataSet_real_prob, (Ndepth, Naz))
    dataSet_class_prob = classifier.predict(dataSet)
    dataSet_class_prob = np.reshape(dataSet_class_prob, (Ndepth, Naz, n_class))
    dataSet_class = np.argmax(dataSet_class_prob, axis=-1)
    # extract the feature
    d_input = discriminator.get_layer(index=0).input
    d_feature = discriminator.get_layer(name='d_feature').output
    d_feature = GlobalAveragePooling2D()(d_feature)
    feature_extractor = Model(d_input, d_feature)
    dataSet_feature = feature_extractor.predict(dataSet)
    db = KMeans(n_clusters=n_class, random_state=100).fit(dataSet_feature)
    dataSet_feature_class = db.labels_
    dataSet_feature_class = np.reshape(dataSet_feature_class, (Ndepth, Naz))
    # rotate against RB
    dataSet_real_prob_rot = orient_rb(dataSet_real_prob, -rbmastInput)
    dataSet_class_prob_rot = orient_rb(dataSet_class_prob, -rbmastInput)
    dataSet_class_rot = orient_rb(dataSet_class, -rbmastInput)
    dataSet_feature_class_rot = orient_rb(dataSet_feature_class, -rbmastInput)
    # save labels
    infoganClassFile = '%s_infogan_class_cluster%d_comp.mat' % (save_filename_prefix, n_class)
    sio.savemat(infoganClassFile, mdict={'dep': depInput,
                                         'infoganRealProb': dataSet_real_prob_rot,
                                         'infoganClassProb': dataSet_class_prob_rot,
                                         'infoganClass': dataSet_class_rot,
                                         'infoganFeatureClass': dataSet_feature_class_rot})

    pass