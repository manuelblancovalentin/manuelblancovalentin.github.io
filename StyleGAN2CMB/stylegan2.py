"""."""
""" Basic modules """
import os
from glob import glob
import numpy as np

""" Visualization """
import matplotlib.pyplot as plt
from PIL import Image

""" 
Import DynamicTable 
(https://github.com/manuelblancovalentin/DynamicTable)
"""
from DynamicTable import DynamicTable

""" Tensorflow """
import tensorflow as tf
import tensorflow.keras.backend as K

""" Custom layers & blocks """
from .layers import Conv2DMod, g_block, d_block

""" Custom losses & utils """
from .losses import gradient_penalty
from . import utils

cha = 24

""" Main GAN """
class GAN(object):
    def __init__(self, input_shape, latent_size = 512, lr = 0.0001, decay = 0.00001, output_dir = None):
        """."""
        """ Setup dirs """
        self.output_dir = output_dir if output_dir is not None else os.getcwd()
        self.models_dir = os.path.join(self.output_dir, 'Models')
        self.results_dir = os.path.join(self.output_dir, 'Results')
        """ Make dirs """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        """ Init params """
        self.LR = lr
        self.beta = 0.999
        self.decay = decay
        self.pl_mean = 0
        self.av = np.zeros([44])

        """ Dataset image sizes """
        self.input_shape = input_shape
        self.__in_side__ = self.input_shape[0]
        assert(self.input_shape[0] == self.input_shape[1])
        self.num_layers = int(np.log2(self.input_shape[0]) - 1)
        self.latent_size = latent_size

        """ Init models """
        self.__init_models__()

        """ Setup optimizers """
        self.GMO = tf.keras.optimizers.Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)
        self.DMO = tf.keras.optimizers.Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)

        """ Init losses """
        self.loss = dict(epoch=[], D=[], G=[], PL=[])

    """ Models initializer """
    def __init_models__(self):
        """."""
        """ Discriminator """
        self.D = None
        self.__build_discriminator__()

        """ Generator (and StyleMapping network) """
        self.S = None
        self.G = None
        # Evaluation models
        self.GE = None
        self.SE = None
        self.__build_generator__()

        #self.DM = None
        #self.AM = None

    @property
    def discriminator(self):
        if not self.D:
            """ Build discriminator """
            self.__build_discriminator__()
        return self.D

    """ Discriminator constructor """
    def __build_discriminator__(self):
        """."""
        """ Input layer """
        ip = tf.keras.layers.Input(shape = self.input_shape, name='disc_input')

        """ Series of discriminator blocks """
        x = ip
        for i,r in enumerate([1,2,4,6,8,16,32]):
            x = d_block(x, r*cha, p = i < 6)

        """ Flatten act map to vector """
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, kernel_initializer = 'he_uniform')(x)

        """ Create model """
        self.D = tf.keras.Model(inputs = ip, outputs = x, name='discNet')

    @property
    def generator(self):
        if not self.G:
            self.__build_generator__()
        return self.G

    """ Generator constructor """
    def __build_generator__(self):
        """."""
        """ Style Mapping network """
        self.S = tf.keras.models.Sequential(name='StyleMappingNet')
        kw = {'input_shape': [self.latent_size]}
        for _ in range(4):
            self.S.add(tf.keras.layers.Dense(512,**kw))
            self.S.add(tf.keras.layers.LeakyReLU(0.2))
            """ we only need to define input_shape the first time, so pop it out """
            kw = {}

        """ 
        Generator network 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([512], name = f'input_style{i}') for i in range(self.num_layers)]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'input_noise')

        """ Latent space """
        x = tf.keras.layers.Lambda(lambda x: x[:, :1] * 0 + 1, name=f'latent_style')(ip_styles[0])

        """ First activation """
        x = tf.keras.layers.Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = tf.keras.layers.Reshape([4, 4, 4*cha])(x)

        """ Loop thru outputs before merging them (partial results) """
        partials = []
        for i, (ip_st,d) in enumerate(zip(ip_styles,[16,8,6,4,2,1])):
            x, r = g_block(x, ip_st, ip_noise, d*cha, self.input_shape[-1], self.__in_side__, u = i > 0)
            partials.append(r)

        """ Add partials """
        x = tf.keras.layers.Add()(partials)

        """ Use values centered around 0, but normalize to [0, 1], providing better initialization """
        x = tf.keras.layers.Lambda(lambda y: y/2 + 0.5, name='normalizer')(x)

        """ Build actual model """
        self.G = tf.keras.models.Model(inputs = ip_styles + [ip_noise], outputs = x, name='generator')

        """ Not sure why the author did this after initializing the models """
        self.GE = tf.keras.models.clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = tf.keras.models.clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

        """ Now build models for evaluation """
        self.__build_eval_G__()
        self.__build_param_avg_G__()

    """ Model for evaluation """
    def __build_eval_G__(self):
        """."""
        """ 
        Create inputs and styles 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([self.latent_size], name=f'eva_input_style{i}') for i in range(self.num_layers)]
        styles = [self.S(ip) for ip in ip_styles]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'eva_input_noise')

        """ Connect to normal generator """
        gf = self.G(styles + [ip_noise])

        """ Build actual eva network """
        self.GM = tf.keras.models.Model(inputs=ip_styles + [ip_noise], outputs=gf, name='eva_generator')

    """ Parameter averaged Generator model """
    def __build_param_avg_G__(self):
        """."""
        """ 
        Create inputs and styles 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([self.latent_size], name=f'ema_input_style{i}') for i in
                     range(self.num_layers)]
        styles = [self.SE(ip) for ip in ip_styles]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'ema_input_noise')

        """ Connect to normal generator """
        gf = self.GE(styles + [ip_noise])

        """ Build actual eva network """
        self.GMA = tf.keras.models.Model(inputs=ip_styles + [ip_noise], outputs=gf, name='ema_generator')

    """ Parameter averaging method """
    def EMA(self):
        #Parameter Averaging

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    """ Reinitialize MA: Reset Parameter Averaging """
    def MAinit(self):
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())

    """ Create noise method """
    def create_noise(self, *args, list = False):
        if not list:
            return utils.noise(*args, self.latent_size)
        else:
            if list == 'mixed':
                return utils.mixedList(*args, self.num_layers, self.latent_size)
            else:
                return utils.noiseList(*args, self.num_layers, self.latent_size)

""" StyleGAN2 model """
class StyleGAN(GAN):
    def __init__(self, input_shape, channel_names = None, **kwargs):
        """."""
        """ Call super (create GAN and models) """
        super().__init__(input_shape, **kwargs)

        """ Define channel names """
        self.channel_names = channel_names if channel_names is not None else [f'channel{i}' for i in range(input_shape[-1])]

    """ Train method """
    def train(self, dataset, epochs = 1e6, batch_size = 16, mixed_prob = 0.9, silent = True):
        """."""
        """ Init losses log """
        losses_log_file = os.path.join(self.output_dir,'losses')
        with open(losses_log_file, 'w') as fh:
            fh.write('epoch,D,G,PL')

        """ 
            Setup DynamicTable 
        """
        header = ['Epoch', 'Progress', 'D_loss', 'G_loss', 'PL']
        formatters = {'Epoch': '{:05d}', 'Progress': '%$',
                      'D_loss': '{:.8f}', 'G_loss': '{:.8f}', 'PL': '{:1.8e}'}
        progress_table = DynamicTable(header, formatters)

        """ Loop thru epochs """
        for e in range(int(epochs)):

            #Train Alternating
            if np.random.random() < mixed_prob:
                style = self.create_noise(batch_size, list='mixed')
            else:
                style = self.create_noise(batch_size, list=True)

            #Apply penalties every 16 steps
            apply_gradient_penalty = e % 2 == 0 or e < 10000
            apply_path_penalty = e % 16 == 0

            """ Get batch from dataset """
            imgs = dataset.get_batch(batch_size).astype('float32')

            """ Apply training step """
            a, b, c, d = self.train_step(imgs, style,
                                         utils.noise_image(batch_size, self.input_shape),
                                         apply_gradient_penalty,
                                         apply_path_penalty)

            #Adjust path length penalty mean
            #d = pl_mean when no penalty is applied
            if self.pl_mean == 0:
                self.pl_mean = np.mean(d)
            self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(d)

            if e % 10 == 0 and e > 20000:
                self.EMA()

            if e <= 25000 and e % 1000 == 2:
                self.MAinit()

            if np.isnan(a):
                print("NaN Value Error.")
                exit()

            """ Append losses """
            self.loss['epoch'].append(e)
            self.loss['D'].append(a.numpy())
            self.loss['G'].append(b.numpy())
            self.loss['PL'].append(self.pl_mean)

            if (e%100 == 0) and not silent:

                #print("\n\nRound " + str(e) + ":")
                #print("D:", np.array(a))
                #print("G:", np.array(b))
                #print("PL:", self.pl_mean)

                #s = round((time.time() - self.lastblip), 4)
                #self.lastblip = time.time()

                #steps_per_second = 100 / s
                #steps_per_minute = steps_per_second * 60
                #steps_per_hour = steps_per_minute * 60
                #print("Steps/Second: " + str(round(steps_per_second, 2)))
                #print("Steps/Hour: " + str(round(steps_per_hour)))

                #min1k = floor(1000/steps_per_minute)
                #sec1k = floor(1000/steps_per_second) % 60
                #print("1k Steps: " + str(min1k) + ":" + str(sec1k))
                #steps_left = epochs - 1 - e + 1e-7
                #hours_left = steps_left // steps_per_hour
                #minutes_left = (steps_left // steps_per_minute) % 60

                #print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
                #print()

                """ Save model """
                if e % 500 == 0:
                    self.save(np.floor(e / 100))

                """ Generate outputs and log losses """
                if e % 1000 == 0 or (e % 100 == 0 and e < 2500):

                    """ Generate """
                    self.evaluate(np.floor(e / 100), dataset = dataset)

                    """ Plot losses """
                    fig, axs = plt.subplots(2,1)
                    axs[0].semilogy(self.loss['epoch'],self.loss['D'],label='D')
                    axs[0].semilogy(self.loss['epoch'],self.loss['G'],label='G')
                    axs[1].plot(self.loss['epoch'],self.loss['PL'],label='PL')
                    axs[0].legend()
                    axs[1].legend()
                    fig.savefig(losses_log_file + '.png')
                    plt.close(fig)

                    """ Append losses to log file """
                    with open(losses_log_file,'a+') as fh:
                        fh.write(f"{self.loss['epoch'][-1]},"
                                 f"{self.loss['D'][-1]},"
                                 f"{self.loss['G'][-1]},"
                                 f"{self.loss['PL'][-1]}\n")

            """ Get updated values to be set into table """
            vals = {'Epoch': e, 'Progress': (e % 100) / 100, 'D_loss': a.numpy(), 'G_loss': b.numpy(),
                    'PL': self.pl_mean}

            """ Update and print line """
            if (e%1000 == 0):
                """ Print header """
                progress_table.print_header()
            progress_table.update_line(vals, append=((e % 100) == 99 or e == 0), print=True)

        """ As we exit the loop, print the bottom of the table """
        progress_table.print_bottom()


    @tf.function
    def train_step(self, images, style, noise, perform_gp = True, perform_pl = False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #Get style information
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(style)):
                w_space.append(self.S(style[i]))

            #Generate images
            generated_images = self.G(w_space + [noise])

            #Discriminate
            real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=True)

            #Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                #R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                #Slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (K.std(w_space[i], axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                #Generate from slightly adjusted W space
                pl_images = self.G(w_space_2 + [noise])

                #Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis = [1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        #Get gradients for respective areas
        gradients_of_generator = gen_tape.gradient(gen_loss, self.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        #Apply gradients
        self.GMO.apply_gradients(zip(gradients_of_generator, self.GM.trainable_variables))
        self.DMO.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    """ Summary """
    def summary(self, show_shapes = True, show_layer_names = True, **kwargs):
        """."""

        """
            PRINT GENERATOR
        """
        with open(os.path.join(self.output_dir,'generator'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.G.summary(print_fn=lambda x: fh.write(x + '\n'))

        """ Print model structure (png) """
        tf.keras.utils.plot_model(self.G,
                                  to_file = os.path.join(self.output_dir,'generator.png'),
                                  show_shapes=show_shapes,
                                  show_layer_names=show_layer_names,
                                  **kwargs)

        with open(os.path.join(self.output_dir,'discriminator'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.D.summary(print_fn=lambda x: fh.write(x + '\n'))

        """ Print model structure (png) """
        tf.keras.utils.plot_model(self.D,
                                  to_file=os.path.join(self.output_dir, 'discriminator.png'),
                                  show_shapes=show_shapes,
                                  show_layer_names=show_layer_names,
                                  **kwargs)


    """ Evaluate function """
    def evaluate(self, num = 0, batch_size = 16, dataset = None, trunc = 1.0):

        """ Define output file standard name """
        filename = os.path.join(self.results_dir, f'${int(np.floor(num)):05d}.png')

        """ Create noise """
        n1 = self.create_noise(64, list=True)
        n2 = utils.noise_image(64, self.input_shape)
        #trunc = np.ones([64, 1]) * trunc

        """ Generate images """
        generated_images = self.GM.predict(n1 + [n2], batch_size = batch_size)

        """ Spectra """
        if dataset is not None:
            """ Get scopes """
            spectra = {zn: dataset.spectra[zn].mean(0) for zn in dataset.spectra}

            """ get spectra """
            sp = {zn: dataset.scopes[zn].get_spectra(generated_images[:,:,:,izn:izn+1]) \
                  for izn,zn in enumerate(self.channel_names) if zn in dataset.scopes}

            fig, axs = plt.subplots(len(spectra),1, figsize=(6,3*len(spectra)))
            for ik,k in enumerate(sp):

                color = next(axs[ik]._get_lines.prop_cycler)['color']

                mu, sg = dataset.spectra[k].mean(0), dataset.spectra[k].std(0)
                axs[ik].plot(dataset.scopes[k].cbins, spectra[k], color = 'k', label='true', linewidth=1)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu+sg, mu-sg, color = color, edgecolor = 'none', alpha = .5)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu+2*sg, mu-2*sg, color = color, edgecolor = 'none', alpha = .3)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu+3*sg, mu-3*sg, color = color, edgecolor = 'none', alpha = .1)

                color = next(axs[ik]._get_lines.prop_cycler)['color']

                mu, sg = sp[k].mean(0), sp[k].std(0)
                axs[ik].plot(dataset.scopes[k].cbins, mu, color = color, label='fake', linewidth=1)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + sg, mu - sg, color=color, edgecolor='none', alpha=.5)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 2 * sg, mu - 2 * sg, color=color, edgecolor='none',
                                     alpha=.3)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 3 * sg, mu - 3 * sg, color=color, edgecolor='none',
                                     alpha=.1)
                axs[ik].legend()
                axs[ik].set_title(k)
            plt.tight_layout()
            fig.savefig(filename.replace('$','spectra_'))
            plt.close(fig)

        """ Split images into 8x8 grid """
        r = [np.concatenate(generated_images[i:i+8], axis = 1) for i in range(0, 64, 8)]
        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:,:,c:c+1]) * 255))
                x.save(filename.replace('$', f'i_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1)*255))
            x.save(filename.replace('$','i_'))

        """ Moving Average """
        generated_images = self.GMA.predict(n1 + [n2], batch_size=batch_size)
        #generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size = BATCH_SIZE)
        #generated_images = self.generateTruncated(n1, trunc = trunc)

        r = [np.concatenate(generated_images[i:i+8], axis = 1) for i in range(0, 64, 8)]
        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                x.save(filename.replace('$', f'ema_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
            x.save(filename.replace('$', 'ema_'))

        """ Mixing regularities """
        nn = self.create_noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis = 0)
        tt = int(self.num_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (self.num_layers - tt)

        latent = p1 + [] + p2

        generated_images = self.GMA.predict(latent + [utils.noise_image(64, self.input_shape)], batch_size = batch_size)
        #generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size = BATCH_SIZE)
        #generated_images = self.generateTruncated(latent, trunc = trunc)

        r = [np.concatenate(generated_images[i:i+8], axis = 0) for i in range(0,64,8)]
        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                x.save(filename.replace('$', f'mr_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
            x.save(filename.replace('$', 'mr_'))


    def generateTruncated(self, style, noi = np.zeros([44]), batch_size = 16, trunc = 0.5, outImage = False, num = 0):

        #Get W's center of mass
        if self.av.shape[0] == 44: #44 is an arbitrary value
            print("Approximating W center of mass")
            self.av = np.mean(self.S.predict(self.create_noise(2000), batch_size = 64), axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        if noi.shape[0] == 44:
            noi = utils.noise_image(64, self.input_shape)

        w_space = []
        pl_lengths = self.pl_mean
        for i in range(len(style)):
            tempStyle = self.S.predict(style[i])
            tempStyle = trunc * (tempStyle - self.av) + self.av
            w_space.append(tempStyle)

        generated_images = self.GE.predict(w_space + [noi], batch_size = batch_size)

        if outImage:
            filename = os.path.join(self.results_dir, f't{int(np.floor(num)):05d}$.png')

            r = []

            for i in range(0, 64, 8):
                r.append(np.concatenate(generated_images[i:i+8], axis = 0))

            c1 = np.concatenate(r, axis = 1)
            c1 = np.clip(c1, 0.0, 1.0)

            """ Apply colormap if single channel"""
            if c1.ndim == 3 and c1.shape[-1] > 1:
                """ Split """
                for c in range(c1.shape[2]):
                    x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                    x.save(filename.replace('$', f'_{self.channel_names[c]}'))
            else:
                x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
                x.save(filename.replace('$', ''))


        return generated_images

    """ Save model method """
    def saveModel(self, model, name, num):
        json = model.to_json()

        with open(os.path.join(self.models_dir, f"{name}.json"), "w") as json_file:
            json_file.write(json)

        model.save_weights(os.path.join(self.models_dir, f'{name}_{int(np.floor(num)):05d}.h5'))

    """ Load model """
    def loadModel(self, name, num):

        json_file = os.path.join(self.models_dir, f'{name}.json')
        weights_file = os.path.join(self.models_dir, f'{name}_{int(np.floor(num)):05d}.h5')
        file = open(json_file, 'r')
        json = file.read()
        file.close()

        loaded_model = tf.keras.models.model_from_json(json, custom_objects = {'Conv2DMod': Conv2DMod})
        loaded_model.load_weights(weights_file)

        return loaded_model

    """ Save model """
    def save(self, num):
        self.saveModel(self.S, "sty", num)
        self.saveModel(self.G, "gen", num)
        self.saveModel(self.D, "dis", num)

        self.saveModel(self.GE, "genMA", num)
        self.saveModel(self.SE, "styMA", num)


    """ Load model from disk """
    def load(self, num):

        if num < 0:
            print(f'[INFO] - Looking for the latest model in dir...', end='')
            """ Get latest """
            weights_file = {wn: [ww.split(os.sep)[-1].replace(f'{wn}_','').split('.')[-2] for ww in glob(os.path.join(self.models_dir, f'{wn}_*.h5'))] \
                            for wn in ['sty','gen','dis','genMA','styMA']}
            nums = {wn: np.array(weights_file[wn], dtype='int') for wn in weights_file}

            """ Get minimum """
            num = np.min([np.max(nums[xn]) for xn in nums])
            print(f'Found! Loading model number {num}')

        """ Assert this index exists for all models """
        if not all([num in nums[xn] for xn in nums]):
            return False

        #Load Models
        self.D = self.loadModel("dis", num)
        self.S = self.loadModel("sty", num)
        self.G = self.loadModel("gen", num)

        self.GE = self.loadModel("genMA", num)
        self.SE = self.loadModel("styMA", num)

        return True

        #self.GenModel()
        #self.GenModelA()



""" Im2Im network """
class AdvTranslationNet(object):
    def __init__(self, input_shape, output_shape, channel_names = None, latent_size = 512, lr = 0.0001, decay = 0.00001, output_dir = None, **kwargs):
        """."""
        super().__init__()

        """ Setup dirs """
        self.output_dir = output_dir if output_dir is not None else os.getcwd()
        self.models_dir = os.path.join(self.output_dir, 'Models')
        self.results_dir = os.path.join(self.output_dir, 'Results')
        """ Make dirs """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        """ Init params """
        self.LR = lr
        self.beta = 0.999
        self.decay = decay
        self.pl_mean = 0
        self.av = np.zeros([44])

        """ Dataset image sizes """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.__in_side__ = self.input_shape[0]
        assert(self.input_shape[0] == self.input_shape[1])
        assert (self.output_shape[0] == self.output_shape[1])
        self.num_layers = int(np.log2(self.input_shape[0]) - 1)
        self.latent_size = latent_size

        """ Define channel names """
        self.channel_names = {'in': channel_names['in'] if channel_names is not None else [f'channel{i}' for i in
                                                                              range(input_shape[-1])],
                              'out': channel_names['out'] if channel_names is not None else [f'channel{i}' for i in
                                                                                     range(output_shape[-1])]                              }

        """ Init models """
        self.__init_models__()

        """ Setup optimizers """
        self.GMO = tf.keras.optimizers.Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)
        self.DMO = tf.keras.optimizers.Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)

        """ Init losses """
        self.loss = dict(epoch=[], D=[], G=[], PL=[])

    """ Models initializer """
    def __init_models__(self):
        """."""
        """ Discriminator """
        self.D = None
        self.__build_discriminator__()

        """ Translator / encoder decoder (and StyleMapping network) """
        self.S = None
        self.E = None
        self.T = None
        # Evaluation models
        self.TE = None
        self.EE = None
        self.SE = None
        self.__build_transformationNet__()

        #self.DM = None
        #self.AM = None

    @property
    def discriminator(self):
        if not self.D:
            """ Build discriminator """
            self.__build_discriminator__()
        return self.D

    """ Discriminator constructor """
    def __build_discriminator__(self):
        """."""
        """ Input layer """
        ip = tf.keras.layers.Input(shape = self.output_shape, name='disc_input')

        """ Series of discriminator blocks """
        x = ip
        for i,r in enumerate([1,2,4,6,8,16,32]):
            x = d_block(x, r*cha, p = i < 6)

        """ Flatten act map to vector """
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, kernel_initializer = 'he_uniform')(x)

        """ Create model """
        self.D = tf.keras.Model(inputs = ip, outputs = x, name='discNet')

    @property
    def generator(self):
        if not self.T:
            self.__build_transformationNet__()
        return self.T

    """ Transformation net constructor """
    def __build_transformationNet__(self):
        """."""
        """ First of all encoder """
        """ Input layer """
        ip = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')

        """ Series of discriminator blocks """
        x = ip
        for i, r in enumerate([1, 2, 4, 6, 8, 16, 32]):
            x = d_block(x, r * cha, p=i < 6)

        """ Flatten act map to vector """
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_size, kernel_initializer='he_uniform')(x)

        """ Create model """
        self.E = tf.keras.Model(inputs=ip, outputs=x, name='encoder')

        """ Style Mapping network """
        self.S = tf.keras.models.Sequential(name='StyleMappingNet')
        kw = {'input_shape': [self.latent_size]}
        for _ in range(4):
            self.S.add(tf.keras.layers.Dense(512,**kw))
            self.S.add(tf.keras.layers.LeakyReLU(0.2))
            """ we only need to define input_shape the first time, so pop it out """
            kw = {}

        """ 
        Decoder network 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([512], name = f'input_style{i}') for i in range(self.num_layers)]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'input_noise')

        """ Latent space """
        x = tf.keras.layers.Lambda(lambda x: x[:, :1] * 0 + 1, name=f'latent_style')(ip_styles[0])

        """ First activation """
        x = tf.keras.layers.Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = tf.keras.layers.Reshape([4, 4, 4*cha])(x)

        """ Loop thru outputs before merging them (partial results) """
        partials = []
        for i, (ip_st,d) in enumerate(zip(ip_styles,[16,8,6,4,2,1])):
            x, r = g_block(x, ip_st, ip_noise, d*cha, self.output_shape[-1], self.output_shape[1], u = i > 0)
            partials.append(r)

        """ Add partials """
        x = tf.keras.layers.Add()(partials)

        """ Use values centered around 0, but normalize to [0, 1], providing better initialization """
        x = tf.keras.layers.Lambda(lambda y: y/2 + 0.5, name='normalizer')(x)

        """ Build actual model """
        self.G = tf.keras.models.Model(inputs = ip_styles + [ip_noise], outputs = x, name='decoder')

        """ Not sure why the author did this after initializing the models """
        self.GE = tf.keras.models.clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = tf.keras.models.clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

        """ Now build models for evaluation """
        self.__build_eval_G__()
        self.__build_param_avg_G__()

    """ Model for evaluation """
    def __build_eval_G__(self):
        """."""
        """ 
        Create inputs and styles 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([self.latent_size], name=f'eva_input_style{i}') for i in range(self.num_layers)]
        styles = [self.S(ip) for ip in ip_styles]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'eva_input_noise')

        """ Connect to normal generator """
        gf = self.G(styles + [ip_noise])

        """ Build actual eva network """
        self.GM = tf.keras.models.Model(inputs=ip_styles + [ip_noise], outputs=gf, name='eva_generator')

    """ Parameter averaged Generator model """
    def __build_param_avg_G__(self):
        """."""
        """ 
        Create inputs and styles 
        """
        """ Input styles """
        ip_styles = [tf.keras.layers.Input([self.latent_size], name=f'ema_input_style{i}') for i in
                     range(self.num_layers)]
        styles = [self.SE(ip) for ip in ip_styles]
        """ Input noise """
        ip_noise = tf.keras.layers.Input(self.input_shape, name=f'ema_input_noise')

        """ Connect to normal generator """
        gf = self.GE(styles + [ip_noise])

        """ Build actual eva network """
        self.GMA = tf.keras.models.Model(inputs=ip_styles + [ip_noise], outputs=gf, name='ema_generator')

    """ Parameter averaging method """
    def EMA(self):
        #Parameter Averaging

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    """ Reinitialize MA: Reset Parameter Averaging """
    def MAinit(self):
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())

    """ Create noise method """
    def create_noise(self, *args, list = False):
        if not list:
            return utils.noise(*args, self.latent_size)
        else:
            if list == 'mixed':
                return utils.mixedList(*args, self.num_layers, self.latent_size)
            else:
                return utils.noiseList(*args, self.num_layers, self.latent_size)

    """ Train method """

    def train(self, dataset, epochs=1e6, batch_size=16, mixed_prob=0.9, silent=True):
        """."""
        """ Init losses log """
        losses_log_file = os.path.join(self.output_dir, 'losses')
        with open(losses_log_file, 'w') as fh:
            fh.write('epoch,D,G,PL')

        """ 
            Setup DynamicTable 
        """
        header = ['Epoch', 'Progress', 'D_loss', 'T_D_loss', 'T_MAE_loss', 'PL']
        formatters = {'Epoch': '{:05d}', 'Progress': '%$',
                      'D_loss': '{:.8f}', 'T_D_loss': '{:.8f}', 'T_MAE_loss': '{:.8f}', 'PL': '{:1.8e}'}
        progress_table = DynamicTable(header, formatters)

        """ Loop thru epochs """
        for e in range(int(epochs)):

            # Train Alternating
            if np.random.random() < mixed_prob:
                style = self.create_noise(batch_size, list='mixed')
            else:
                style = self.create_noise(batch_size, list=True)

            # Apply penalties every 16 steps
            apply_gradient_penalty = e % 2 == 0 or e < 10000
            apply_path_penalty = e % 16 == 0

            """ Get batch from dataset """
            imgs = dataset.get_batch(batch_size).astype('float32')

            """ Apply training step """
            a, b, c, d = self.train_step(imgs, style,
                                         utils.noise_image(batch_size, self.input_shape),
                                         apply_gradient_penalty,
                                         apply_path_penalty)

            # Adjust path length penalty mean
            # d = pl_mean when no penalty is applied
            if self.pl_mean == 0:
                self.pl_mean = np.mean(d)
            self.pl_mean = 0.99 * self.pl_mean + 0.01 * np.mean(d)

            if e % 10 == 0 and e > 20000:
                self.EMA()

            if e <= 25000 and e % 1000 == 2:
                self.MAinit()

            if np.isnan(a):
                print("NaN Value Error.")
                exit()

            """ Append losses """
            self.loss['epoch'].append(e)
            self.loss['D'].append(a.numpy())
            self.loss['G'].append(b.numpy())
            self.loss['PL'].append(self.pl_mean)

            if (e % 100 == 0) and not silent:

                # print("\n\nRound " + str(e) + ":")
                # print("D:", np.array(a))
                # print("G:", np.array(b))
                # print("PL:", self.pl_mean)

                # s = round((time.time() - self.lastblip), 4)
                # self.lastblip = time.time()

                # steps_per_second = 100 / s
                # steps_per_minute = steps_per_second * 60
                # steps_per_hour = steps_per_minute * 60
                # print("Steps/Second: " + str(round(steps_per_second, 2)))
                # print("Steps/Hour: " + str(round(steps_per_hour)))

                # min1k = floor(1000/steps_per_minute)
                # sec1k = floor(1000/steps_per_second) % 60
                # print("1k Steps: " + str(min1k) + ":" + str(sec1k))
                # steps_left = epochs - 1 - e + 1e-7
                # hours_left = steps_left // steps_per_hour
                # minutes_left = (steps_left // steps_per_minute) % 60

                # print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
                # print()

                """ Save model """
                if e % 500 == 0:
                    self.save(np.floor(e / 100))

                """ Generate outputs and log losses """
                if e % 1000 == 0 or (e % 100 == 0 and e < 2500):
                    """ Generate """
                    self.evaluate(np.floor(e / 100), dataset=dataset)

                    """ Plot losses """
                    fig, axs = plt.subplots(2, 1)
                    axs[0].semilogy(self.loss['epoch'], self.loss['D'], label='D')
                    axs[0].semilogy(self.loss['epoch'], self.loss['G'], label='G')
                    axs[1].plot(self.loss['epoch'], self.loss['PL'], label='PL')
                    axs[0].legend()
                    axs[1].legend()
                    fig.savefig(losses_log_file + '.png')
                    plt.close(fig)

                    """ Append losses to log file """
                    with open(losses_log_file, 'a+') as fh:
                        fh.write(f"{self.loss['epoch'][-1]},"
                                 f"{self.loss['D'][-1]},"
                                 f"{self.loss['G'][-1]},"
                                 f"{self.loss['PL'][-1]}\n")

            """ Get updated values to be set into table """
            vals = {'Epoch': e, 'Progress': (e % 100) / 100, 'D_loss': a.numpy(), 'G_loss': b.numpy(),
                    'PL': self.pl_mean}

            """ Update and print line """
            if (e % 1000 == 0):
                """ Print header """
                progress_table.print_header()
            progress_table.update_line(vals, append=((e % 100) == 99 or e == 0), print=True)

        """ As we exit the loop, print the bottom of the table """
        progress_table.print_bottom()

    @tf.function
    def train_step(self, images, style, noise, perform_gp=True, perform_pl=False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Get style information
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(style)):
                w_space.append(self.S(style[i]))

            # Generate images
            generated_images = self.G(w_space + [noise])

            # Discriminate
            real_output = self.D(images, training=True)
            fake_output = self.D(generated_images, training=True)

            # Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                # R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                # Slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (K.std(w_space[i], axis=0, keepdims=True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                # Generate from slightly adjusted W space
                pl_images = self.G(w_space_2 + [noise])

                # Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis=[1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        # Get gradients for respective areas
        gradients_of_generator = gen_tape.gradient(gen_loss, self.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        # Apply gradients
        self.GMO.apply_gradients(zip(gradients_of_generator, self.GM.trainable_variables))
        self.DMO.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    """ Summary """

    def summary(self, show_shapes=True, show_layer_names=True, **kwargs):
        """."""

        """
            PRINT GENERATOR
        """
        with open(os.path.join(self.output_dir, 'encoder'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.E.summary(print_fn=lambda x: fh.write(x + '\n'))

        """ Print model structure (png) """
        tf.keras.utils.plot_model(self.E,
                                  to_file=os.path.join(self.output_dir, 'encoder.png'),
                                  show_shapes=show_shapes,
                                  show_layer_names=show_layer_names,
                                  **kwargs)

        with open(os.path.join(self.output_dir, 'discriminator'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.D.summary(print_fn=lambda x: fh.write(x + '\n'))

        """ Print model structure (png) """
        tf.keras.utils.plot_model(self.D,
                                  to_file=os.path.join(self.output_dir, 'discriminator.png'),
                                  show_shapes=show_shapes,
                                  show_layer_names=show_layer_names,
                                  **kwargs)

    """ Evaluate function """
    def evaluate(self, num=0, batch_size=16, dataset=None, trunc=1.0):

        """ Define output file standard name """
        filename = os.path.join(self.results_dir, f'${int(np.floor(num)):05d}.png')

        """ Create noise """
        n1 = self.create_noise(64, list=True)
        n2 = utils.noise_image(64, self.input_shape)
        # trunc = np.ones([64, 1]) * trunc

        """ Generate images """
        generated_images = self.GM.predict(n1 + [n2], batch_size=batch_size)

        """ Spectra """
        if dataset is not None:
            """ Get scopes """
            spectra = {zn: dataset.spectra[zn].mean(0) for zn in dataset.spectra}

            """ get spectra """
            sp = {zn: dataset.scopes[zn].get_spectra(generated_images[:, :, :, izn:izn + 1]) \
                  for izn, zn in enumerate(self.channel_names) if zn in dataset.scopes}

            fig, axs = plt.subplots(len(spectra), 1, figsize=(6, 3 * len(spectra)))
            for ik, k in enumerate(sp):
                color = next(axs[ik]._get_lines.prop_cycler)['color']

                mu, sg = dataset.spectra[k].mean(0), dataset.spectra[k].std(0)
                axs[ik].plot(dataset.scopes[k].cbins, spectra[k], color='k', label='true', linewidth=1)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + sg, mu - sg, color=color, edgecolor='none', alpha=.5)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 2 * sg, mu - 2 * sg, color=color, edgecolor='none',
                                     alpha=.3)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 3 * sg, mu - 3 * sg, color=color, edgecolor='none',
                                     alpha=.1)

                color = next(axs[ik]._get_lines.prop_cycler)['color']

                mu, sg = sp[k].mean(0), sp[k].std(0)
                axs[ik].plot(dataset.scopes[k].cbins, mu, color=color, label='fake', linewidth=1)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + sg, mu - sg, color=color, edgecolor='none', alpha=.5)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 2 * sg, mu - 2 * sg, color=color, edgecolor='none',
                                     alpha=.3)
                axs[ik].fill_between(dataset.scopes[k].cbins, mu + 3 * sg, mu - 3 * sg, color=color, edgecolor='none',
                                     alpha=.1)
                axs[ik].legend()
                axs[ik].set_title(k)
            plt.tight_layout()
            fig.savefig(filename.replace('$', 'spectra_'))
            plt.close(fig)

        """ Split images into 8x8 grid """
        r = [np.concatenate(generated_images[i:i + 8], axis=1) for i in range(0, 64, 8)]
        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                x.save(filename.replace('$', f'i_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
            x.save(filename.replace('$', 'i_'))

        """ Moving Average """
        generated_images = self.GMA.predict(n1 + [n2], batch_size=batch_size)
        # generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size = BATCH_SIZE)
        # generated_images = self.generateTruncated(n1, trunc = trunc)

        r = [np.concatenate(generated_images[i:i + 8], axis=1) for i in range(0, 64, 8)]
        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                x.save(filename.replace('$', f'ema_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
            x.save(filename.replace('$', 'ema_'))

        """ Mixing regularities """
        nn = self.create_noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis=0)
        tt = int(self.num_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (self.num_layers - tt)

        latent = p1 + [] + p2

        generated_images = self.GMA.predict(latent + [utils.noise_image(64, self.input_shape)], batch_size=batch_size)
        # generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size = BATCH_SIZE)
        # generated_images = self.generateTruncated(latent, trunc = trunc)

        r = [np.concatenate(generated_images[i:i + 8], axis=0) for i in range(0, 64, 8)]
        c1 = np.concatenate(r, axis=1)
        c1 = np.clip(c1, 0.0, 1.0)

        """ Apply colormap if single channel"""
        if c1.ndim == 3 and c1.shape[-1] > 1:
            """ Split """
            for c in range(c1.shape[2]):
                x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
                x.save(filename.replace('$', f'mr_{self.channel_names[c]}'))
        else:
            x = Image.fromarray(np.uint8(utils.colmap(c1) * 255))
            x.save(filename.replace('$', 'mr_'))

    """ Save model method """
    def saveModel(self, model, name, num):
        json = model.to_json()

        with open(os.path.join(self.models_dir, f"{name}.json"), "w") as json_file:
            json_file.write(json)

        model.save_weights(os.path.join(self.models_dir, f'{name}_{int(np.floor(num)):05d}.h5'))

    """ Load model """
    def loadModel(self, name, num):

        json_file = os.path.join(self.models_dir, f'{name}.json')
        weights_file = os.path.join(self.models_dir, f'{name}_{int(np.floor(num)):05d}.h5')
        file = open(json_file, 'r')
        json = file.read()
        file.close()

        loaded_model = tf.keras.models.model_from_json(json, custom_objects={'Conv2DMod': Conv2DMod})
        loaded_model.load_weights(weights_file)

        return loaded_model

    """ Save model """
    def save(self, num):
        self.saveModel(self.S, "sty", num)
        self.saveModel(self.G, "gen", num)
        self.saveModel(self.D, "dis", num)

        self.saveModel(self.GE, "genMA", num)
        self.saveModel(self.SE, "styMA", num)

    """ Load model from disk """
    def load(self, num):

        if num < 0:
            print(f'[INFO] - Looking for the latest model in dir...', end='')
            """ Get latest """
            weights_file = {wn: [ww.split(os.sep)[-1].replace(f'{wn}_', '').split('.')[-2] for ww in
                                 glob(os.path.join(self.models_dir, f'{wn}_*.h5'))] \
                            for wn in ['sty', 'gen', 'dis', 'genMA', 'styMA']}
            nums = {wn: np.array(weights_file[wn], dtype='int') for wn in weights_file}

            """ Get minimum """
            num = np.min([np.max(nums[xn]) for xn in nums])
            print(f'Found! Loading model number {num}')

        """ Assert this index exists for all models """
        if not all([num in nums[xn] for xn in nums]):
            return False

        # Load Models
        self.D = self.loadModel("dis", num)
        self.S = self.loadModel("sty", num)
        self.G = self.loadModel("gen", num)

        self.GE = self.loadModel("genMA", num)
        self.SE = self.loadModel("styMA", num)

        return True