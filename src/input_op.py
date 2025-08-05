import Input_loader
#from IPython.display import display, Markdown
from rich.console import Console
from rich.markdown import Markdown

console = Console()
x = Input_loader.load_data(dir='/home/nandhan/test', api_custom_model=True)
md = Markdown(x.process())

#console.print(md)
try:
    console.print(md)
except Exception as e:
    print(f"Exception occuring {e}")