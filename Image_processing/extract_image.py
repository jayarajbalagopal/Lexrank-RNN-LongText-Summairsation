from PIL import Image
from pytesseract import image_to_string

x=Image.open('samp.jpg','r')
x=x.convert('L')
x.save('text_black.jpg')

print image_to_string(Image.open('text_black.jpg'))


