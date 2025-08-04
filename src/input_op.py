import Input_loader
from IPython.display import display

x = Input_loader.load_data(dir='/home/nandhan/test', api_from_base=True)
display(x.process())
