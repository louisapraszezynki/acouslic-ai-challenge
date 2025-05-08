import SimpleITK as sitk
import numpy as np
from PIL import Image

ds = sitk.ReadImage("acouslic-ai-train-set/masks/stacked_fetal_abdomen/0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06.mha")

# (1)Convert the SimpleITK image to a numpy array
np_array = sitk.GetArrayFromImage(ds)*100
print(np_array)


def show_image(image_index):
    image = np_array[image_index]

    rouge = np.zeros_like(image)
    rouge[image == 200] = 255

    vert = np.zeros_like(image)
    vert[image == 100] = 255

    bleu = np.zeros_like(image)

    image_rgb = np.stack([rouge, vert, bleu], axis=-1)
    Image.fromarray(image_rgb).show()

filtered_array = np_array.sum(axis=-1).sum(axis=-1)
nonzero = filtered_array.nonzero()[0].tolist()

show_image(nonzero[10])


# frame = np_array[48]
# image = Image.fromarray(frame)
# image.show()

# fonction qui prend une matrice de 840 images en entrée
# fonction qui renvoie les indices qui correspondent aux images qui ont un masque 1 ou 2



# show_image(nonzero[10])

# créer des images en fonction des pixel values
# 0 : 0 RGB
# 1 : 255 dans G
# 2 : 255 dans R

# for i in nonzero:
#     np_array[i] =


