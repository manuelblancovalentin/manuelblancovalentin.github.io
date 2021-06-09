"""
    __author__: Manuel Blanco Valentin
"""
""" Main modules """
import os

""" Import main module"""
import StyleGAN2CMB as sgcmb


"""--------------------------------------------------------------------------
PART 0: Preparing the data
--------------------------------------------------------------------------"""
""" Define maps to be used """
maps = ['q','u','k','b','e']

""" Init dataset object """
CMB_dir = "/home/dados4T/DeepCMB/deepskies-cmb/r=0.1_simulations/"
dataset = sgcmb.CMBDataset(CMB_dir, maps = maps, reload=False)


"""--------------------------------------------------------------------------
PART 1: Training generator (Z/W -> q,u,k,e,b)
--------------------------------------------------------------------------"""
""" Init params """
epochs = 1e6
batch_size = 16

""" Create StyleGAN2 Model """
output_dir = os.path.join(os.getcwd(),'results',''.join(maps))

""" Build model """
input_shape = dataset.get_batch(1)[0].shape[1:]
model = sgcmb.StyleGAN(input_shape, channel_names = maps, lr = 0.0001, output_dir = output_dir)

""" Print summary (and png with architectures) """
model.summary()

""" Try to load the previously trained model or train (if not found) """
if not model.load(-1):
    """ Train """
    model.train(dataset, epochs, batch_size = batch_size, silent = False)


"""--------------------------------------------------------------------------
PART 2: Image transformation / Fine tuning (q,u,k,e [DISC*] -> Z/W -> [GEN] -> b)
--------------------------------------------------------------------------"""
""" First of all we must create the transformation network:
        - DISC* is a modified styleGAN2 discriminator which we will use to extract features from input maps
        - The output of DISC* is connected to a GEN model (non-trainable/freezed weights)
        - DISC* is trained to optimize its outputs (Z/W) to mimic the structure of latent space/noise to create the right b maps
        - The output must be compared for match with the original true maps ||b-^b||
"""
output_shape = dataset.get_batch(1)[0][:,:,:,0:1].shape[1:]
output_T_dir = os.path.join(os.getcwd(),'results',''.join(maps),'translation')
T_model = sgcmb.AdvTranslationNet(input_shape[:-1] + (input_shape[-1]-1,),
                                  output_shape,
                                  channel_names = {'in': [m for m in maps if m != 'b'], 'out': ['b']}, lr = 0.0001,
                                  output_dir = output_T_dir)

""" Initialize with previous weights from sgan """
import tensorflow as tf
#T_model.G = tf.keras.models.clone_model(model.G)

for ii,(lyg,lyt) in enumerate(zip(model.G.layers,T_model.G.layers)):

    wg = lyg.weights
    wt = lyt.weights

    if (wg == []) or (wt == []):
        T_model.G.layers[ii].trainable = False
    else:

        try:

            ww = []
            tr = False
            for wwg,wwt in zip(wg,wt):

                if len(wwg) == 0 or len(wwt) == 0 or wwg != wwt:
                    """ Keep new """
                    ww.append(wwt)
                    tr = True
                else:
                    print(f'[INFO] - Setting up pre-trained weight on layer {T_model.G.layers[ii]}')
                    ww.append(wwg)

            T_model.G.layers[ii].set_weights(ww)
            T_model.G.layers[ii].trainable = tr
        except:
            T_model.G.layers[ii].trainable = True



""" Print summary (and png with architectures) """
T_model.summary()

""" Try to load the previously trained model or train (if not found) """
if not T_model.load(-1) or True:
    """ Train """
    T_model.train(dataset, epochs, batch_size = batch_size, silent = False)
else:
    exit()



"""
model.load(31)

n1 = noiseList(64)
n2 = nImage(64)
for i in range(50):
    print(i, end = '\r')
    model.generateTruncated(n1, noi = n2, trunc = i / 50, outImage = True, num = i)
"""
