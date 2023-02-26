from keras.models import load_model
from utils.model import AkmalCNN

model = load_model(
    'model/test_model.h5',
    custom_objects={
        'dice_loss_plus_1focal_loss': AkmalCNN(7, 512, 512, 4).bobot(),
        'jacard_coef': AkmalCNN(7, 512, 512, 4).jacard_coef()
    })
