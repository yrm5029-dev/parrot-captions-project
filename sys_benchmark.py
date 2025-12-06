from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

class sys_text_dataset(Dataset):
    def __init__(self, text_list, transform, style=0):
        self.style_list = [
            ('white', 'black'),
            ('black', 'white'),
            ('gray', 'white'),
            ('gray', 'black'),
        ]
        self.style = self.style_list[style]
        if isinstance(text_list, list):
            text_list = text_list
        else:
            df = pd.read_csv(text_list)
            text_list = df['Unnamed: 0'].tolist()
        print(len(text_list))
        self.transform = transform
        self.text_list = []
        try:
            self.font = ImageFont.truetype("Arial.ttf", 32)
        except Exception:
            # Fallback for systems without Arial
            self.font = ImageFont.load_default()
            print("Warning: Arial.ttf not found, using default font.")
        for t in text_list:
            if self.can_render_string(t):
                self.text_list.append(t)

    def __len__(self):
        return len(self.text_list)

    def set_style(self, style):
        self.style = self.style_list[style]


    def can_render_string(self, text):
        try:
            bbox = self.font.getbbox(text)
            return True
        except Exception as e:
            return False
        

    def draw_text(self, text):
        # Create a 224 x 224 gray image
        img = Image.new('RGB', (224, 224), self.style[0])
        
        # Get the draw object
        draw = ImageDraw.Draw(img)
        # Calculate width and height of the text to be drawn
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Calculate the x, y coordinates of the text
        x = img.width / 2 - text_width / 2
        y = img.height / 2 - text_height / 2
        # Draw the text on the image
        draw.text((x, y), text, font=self.font, fill=self.style[1])
        return img

    def __getitem__(self, index):
        text = self.text_list[index]
        image = self.draw_text(text)
        image = self.transform(image)
        return image, text