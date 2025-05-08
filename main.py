import SimpleITK as sitk
import numpy as np
from PIL import Image

ds = sitk.ReadImage("acouslic-ai-train-set/images/stacked_fetal_ultrasound/0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06.mha")

# Convert the SimpleITK image to a numpy array
np_array = sitk.GetArrayFromImage(ds)

frame = np_array[48]
image = Image.fromarray(frame)
image.show()