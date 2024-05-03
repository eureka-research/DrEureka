# %%

from PIL import Image

def create_tiled_image(image_path, tile_count_x=10, tile_count_y=10):
    # Load the image
    original_image = Image.open(image_path)
    # width, height = original_image.size
    width, height = 3000, 2000

    # Calculate the size of each tile
    tile_width = width // tile_count_x
    tile_height = height // tile_count_y

    # Create a new image with the same resolution as the original
    new_image = Image.new('RGB', (width, height))

    # Iterate through the tiles and paste the resized original image
    for x in range(tile_count_x):
        for y in range(tile_count_y):
            resized_image = original_image.resize((tile_width, tile_height))
            new_image.paste(resized_image, (tile_width * x, tile_height * y))

    # Save the new tiled image
    new_image.save(f'tiled_{image_path}')

if __name__ == '__main__':
    image_paths = ['snow.jpg', 'grass.jpg', 'pebble_stone_texture_nature.jpg', 'particle_board_paint_aged.jpg', 'texture_stone_stone_texture_0.jpg', 'texture_wood_brown_1033760.jpg', 'brick_texture.jpg']
    # image_path = 'ice_texture.jpg'  # Replace with the path to your image (PNG or JPG)
    for image_path in image_paths:
        create_tiled_image(image_path, tile_count_x=5, tile_count_y=5)

# %%
