import numpy as np
from torch.utils.data import Dataset
from .synthetic_shapes_functions import parse_primitives, generate_background, add_gausian_noise, pad_points

drawing_primitives = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube'
]

class mySyntheticShapes(Dataset):
    def __init__(self, config, num_imgs_returned, transform=None):
        self.config = config 
        self.num_imgs_returned = num_imgs_returned
        self.transform = transform

    def __len__(self):
        return self.num_imgs_returned

    def __getitem__(self, index):
        i=0
        while True:
            try:
                primitives = parse_primitives(self.config['primitives'], drawing_primitives)
                primitive = np.random.choice(primitives)

                image = generate_background(self.config['generation']['image_size'], **self.config['generation']['params']['generate_background'])
                image = add_gausian_noise(image)

                primitive_func_str = primitive+'(image, **self.config["generation"]["params"].get(primitive, {}))'
                points = np.array(eval(primitive_func_str))
                points = pad_points(points, desired_num_points=60)

                sample = (image,points)

                if self.transform:
                    sample = self.transform(sample)

            except:
                print("An exception occurred")
                i+=1
                if i<10:
                    continue
                break
        return sample
