from fastai.vision.all import *

learn = load_learner('geometry.pkl')
img = PILImageBW.create('images/star.jpg')
print(learn.predict(img))