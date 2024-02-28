import visualkeras
from nn.conv.lenet import LeNet
from keras.utils import plot_model
from PIL import ImageFont

# need to install graphviz locally using brew or choco as well as in requirements.txt file

model = LeNet.build(28, 28, 1, 10)

graphviz_path = "starter_bundle/19_visualising_network_arc/lenet_arch.png"
plot_model(model, to_file=graphviz_path, show_shapes=True)

vis_path = "starter_bundle/19_visualising_network_arc/lenet_graph.png"
font = ImageFont.truetype("arial.ttf", 11)
visualkeras.layered_view(model=model, to_file=vis_path, legend=True, font=font)

