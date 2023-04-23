import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([2, 2, 2])
points = torch.tensor([[20.0,  30.0], # base
                       [50.0,  60.0], # control point
                       [ 90.0, 198.0], # control point
                       [ 60.0, 218.0], # base
                       [ 90.0, 180.0], # control point
                       [200.0,  85.0], # control point
                       [230.0,  90.0], # base
                       [220.0,  70.0], # control point
                       [130.0,  55.0]]) # control point
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = True)
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/test_triangles/target.png', gamma=2.2)
target = img.clone()

# Load the obj file, get the vertices/control points
obj = "imgs/Triangles.obj"
vertices_tmp, faces_tmp = pydiffvg.obj_to_scene(obj)
print(float(vertices_tmp[1][1]))
print(faces_tmp)

vertices = []
faces = []
for v in vertices_tmp:
    vertices.append([float(v[1]), float(v[2])])
for f in faces_tmp:
    vs_count = len(f)
    tmp = []
    for v in f[1:]:
        tmp.append(int(v))
    faces.append(tmp)
print(vertices)
print(faces)

quit()

# Ternary subdivision
# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[vertices[0][0]/256.0, vertices[0][1]/256.0], # base
                         [70.0/256.0,  140.0/256.0], # control point
                         [50.0/256.0, 100.0/256.0], # control point
                         [vertices[1][0]/256.0, vertices[1][1]/256.0], # base
                         [80.0/256.0, 40.0/256.0], # control point
                         [120.0/256.0,  40.0/256.0], # control point
                         [vertices[2][0]/256.0, vertices[2][1]/256.0], # base
                         [150.0/256.0,  100.0/256.0], # control point
                         [130.0/256.0,  140.0/256.0]], # control point
                        requires_grad = True) 
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
path.points = points_n * 256
path_group.fill_color = color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/test_triangles/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, color], lr=1e-2)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    path.points = points_n * 256
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/test_triangles/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('color.grad:', color.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', path.points)
    print('color:', path_group.fill_color)

# Render the final result.
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None,
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/test_triangles/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_triangles/iter_%d.png", "-vb", "20M",
    "results/test_triangles/out.mp4"])

