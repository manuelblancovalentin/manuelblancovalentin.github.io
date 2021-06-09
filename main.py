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
input_shape = dataset.get_batch(1).shape[1:]
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
output_shape = dataset.get_batch(1)[:,:,:,0:1].shape[1:]
output_T_dir = os.path.join(os.getcwd(),'results',''.join(maps),'translation')
T_model = sgcmb.AdvTranslationNet(input_shape, output_shape, channel_names = maps, lr = 0.0001, output_dir = output_T_dir)

""" Print summary (and png with architectures) """
T_model.summary()



n1 = model.create_noise(64, list=True)
n2 = sgcmb.stylegan2.utils.noise_image(64, input_shape)
#trunc = np.ones([64, 1]) * trunc

for i in range(50):
    print(i, end = '\r')
    model.generateTruncated(n1, noi = n2, trunc = i / 50, outImage = True, num = i)

""" Generate images """
generated_images = model.GM.predict(n1 + [n2], batch_size = batch_size)

""" Split images into 8x8 grid """
r = [np.concatenate(generated_images[i:i+8], axis = 1) for i in range(0, 64, 8)]
c1 = np.concatenate(r, axis = 0)
c1 = np.clip(c1, 0.0, 1.0)

#for c in range(c1.shape[2]):
#x = Image.fromarray(np.uint8(utils.colmap(c1[:, :, c:c + 1]) * 255))
plt.imshow(c1[:,:,0],cmap='bwr'); plt.show()




""" Now use style """
#n1 = model.create_noise(64, list=True)
#n2 = sgcmb.stylegan2.utils.noise_image(64, input_shape)
#trunc = np.ones([64, 1]) * trunc
import numpy as np

n1 = model.create_noise(64, list=True)
n2 = np.random.random((64,) + input_shape)

""" Generate images """
generated_images = model.GM.predict(n1 + [n2], batch_size = batch_size)

""" Split images into 8x8 grid """
r = [np.concatenate(generated_images[i:i+8], axis = 1) for i in range(0, 64, 8)]
c1 = np.concatenate(r, axis = 0)
c1 = np.clip(c1, 0.0, 1.0)

import matplotlib.pyplot as plt
plt.imshow(c1[:,:,0],cmap='bwr'); plt.show()




"""
model.load(31)

n1 = noiseList(64)
n2 = nImage(64)
for i in range(50):
    print(i, end = '\r')
    model.generateTruncated(n1, noi = n2, trunc = i / 50, outImage = True, num = i)
"""
