import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors

# connection between the 8 points of 3d bbox
BONES_3D_BBOX = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
]


def plot_2d_bbox(bbox_2d, bones, color, ax):
    if ax is None:
        axx = plt
    else:
        axx = ax
    colors = cm.rainbow(np.linspace(0, 1, len(bbox_2d)))
    for pt, c in zip(bbox_2d, colors):
        axx.scatter(pt[0], pt[1], color=c, s=50)

    if bones is None:
        bones = BONES_3D_BBOX
    for bone in bones:
        sidx, eidx = bone
        # bottom of bbox is white
        if min(sidx, eidx) >= 4:
            color = "w"
        axx.plot(
            [bbox_2d[sidx][0], bbox_2d[eidx][0]],
            [bbox_2d[sidx][1], bbox_2d[eidx][1]],
            color,
        )
    return axx


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D
    numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image
    in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def concat_pil_images(images):
    """
    Put a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def stack_pil_images(images):
    """
    Stack a list of PIL images next to each other
    """
    assert isinstance(images, list)
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def im_list_to_plt(image_list, figsize, title_list=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=figsize)
    for idx, (ax, im) in enumerate(zip(axes, image_list)):
        ax.imshow(im)
        ax.set_title(title_list[idx])
    fig.tight_layout()
    im = fig2img(fig)
    plt.close()
    return im


def visualize_hand_keypoints(keypoints, image=None, viz_color=None, image_size=(800, 800), draw_legend=False, draw_title=False):
    """
    Visualize 2D hand keypoints following MANO ordering and kinematic chain
    
    Parameters:
    -----------
    keypoints : numpy.ndarray of shape (21, 2)
        The 2D keypoints for hand joints, normalized from 0 to 1
    image : PIL.Image, optional
        Background image (if None, a blank image will be created)
    image_size : tuple, optional
        Size of the output image (width, height) if no input image is provided
        
    Returns:
    --------
    PIL.Image
        Image with visualized hand keypoints
    """
    # Define connections according to MANO kinematic chain
    connections = [
        # Thumb connections
        [0, 13], [13, 14], [14, 15], [15, 16],
        # Index finger connections
        [0, 1], [1, 2], [2, 3], [3, 17],
        # Middle finger connections
        [0, 4], [4, 5], [5, 6], [6, 18],
        # Ring finger connections
        [0, 10], [10, 11], [11, 12], [12, 19],
        # Pinky connections
        [0, 7], [7, 8], [8, 9], [9, 20]
    ]
    
    # Define joint types for coloring
    wrist_joint = [0]
    index_joints = [1, 2, 3, 17]
    middle_joints = [4, 5, 6, 18]
    ring_joints = [10, 11, 12, 19]
    pinky_joints = [7, 8, 9, 20]
    thumb_joints = [13, 14, 15, 16]
    
    # Color mapping
    color_map = {
        'wrist': (255, 0, 0),  # Red
        'thumb': (0, 255, 0),  # Green
        'index': (0, 0, 255),  # Blue
        # 'middle': (0, 255, 255),  # Cyan
        'middle': (255, 127, 0),  # Orange
        'ring': (255, 0, 255),  # Magenta
        # 'pinky': (255, 255, 0)  # Yellow
        'pinky': (127, 0, 255)  # Purple
    }
    
    # Create or use background image
    if image is None:
        # Create a blank white image
        img = Image.new('RGB', image_size, color=(255, 255, 255))
    else:
        # Use the provided image
        img = image.copy()
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Scale normalized keypoints to image dimensions
    keypoints_scaled = []
    for kp in keypoints:
        x = int(kp[0] * width)
        y = int(kp[1] * height)
        keypoints_scaled.append((x, y))
    
    if viz_color is None:
        # line_color = (0, 0, 255) # default to blue
        line_color = (0, 255, 255) # cyan
    else:
        line_color = (255, 165, 0) # orange
        # line_color = (255, 255, 0) # yellow
    # Draw connections (bones)
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(keypoints_scaled) and end_idx < len(keypoints_scaled):
            start_point = keypoints_scaled[start_idx]
            end_point = keypoints_scaled[end_idx]
            draw.line([start_point, end_point], fill=line_color, width=2)
    
    # Draw keypoints (joints)
    for i, point in enumerate(keypoints_scaled):
        if viz_color is None:
            if i in wrist_joint:
                color = color_map['wrist']
            elif i in index_joints:
                color = color_map['index']
            elif i in middle_joints:
                color = color_map['middle']
            elif i in ring_joints:
                color = color_map['ring']
            elif i in pinky_joints:
                color = color_map['pinky']
            elif i in thumb_joints:
                color = color_map['thumb']
            else:
                color = (0, 0, 0)  # Black
        else:
            color = viz_color
        
        # Draw circle for keypoint
        circle_radius = 2 # max(5, int(min(width, height) * 0.01)) / 2
        draw.ellipse((point[0] - circle_radius, point[1] - circle_radius,
                      point[0] + circle_radius, point[1] + circle_radius),
                      fill=color, outline=(0, 0, 0))
        
        # # Add joint index labels
        # # Try to load a font, or use default if not available
        # try:
        #     font = ImageFont.truetype("arial.ttf", size=max(10, int(min(width, height) * 0.015)))
        # except IOError:
        #     font = ImageFont.load_default()
        
        # draw.text((point[0] + circle_radius, point[1] - circle_radius), 
        #          str(i), fill=(0, 0, 0), font=font)
    
    if draw_legend:
        # Add legend
        legend_items = [
            ("0: Wrist", color_map['wrist']),
            ("1-3,17: Index", color_map['index']),
            ("4-6,18: Middle", color_map['middle']),
            ("10-12,19: Ring", color_map['ring']),
            ("7-9,20: Pinky", color_map['pinky']),
            ("13-16: Thumb", color_map['thumb'])
        ]
        
        # Position legend in top-right corner
        legend_x = width - 180
        legend_y = 20
        legend_spacing = 25
        
        # Draw legend background
        legend_width = 160
        legend_height = len(legend_items) * legend_spacing + 10
        draw.rectangle((legend_x - 10, legend_y - 10, 
                        legend_x + legend_width, legend_y + legend_height), 
                    fill=(255, 255, 255, 128), outline=(0, 0, 0))
        
        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y = legend_y + i * legend_spacing
            draw.ellipse((legend_x, y, legend_x + 15, y + 15), fill=color, outline=(0, 0, 0))
            draw.text((legend_x + 25, y), label, fill=(0, 0, 0), font=font)
    
    if draw_title:
        # Add title
        title = "Hand Keypoints (Arctic MANO Ordering)"
        title_font_size = max(12, int(min(width, height) * 0.02))
        try:
            title_font = ImageFont.truetype("arial.ttf", size=title_font_size)
        except IOError:
            title_font = ImageFont.load_default()
        
        title_width = draw.textlength(title, font=title_font)
        draw.text(((width - title_width) // 2, 10), title, fill=(0, 0, 0), font=title_font)
    
    return img