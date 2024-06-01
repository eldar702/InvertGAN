import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

class DatasetManager:
    ARTIST_NAMES = ["bela-czobel", "camille-pissarro"]
    WIKIART_URL = "https://www.wikiart.org/en/{}/all-works/text-list"
    IMAGE_SIZE = 256
    TEST_SPLIT = 0.2

    def get_image_links(self, artist_name, page):
        image_links = []
        artist_url = self.WIKIART_URL.format(artist_name) + f'?json=2&page={page}'
        response = requests.get(artist_url)
        if response.status_code == 200:
            response_json = response.json()
            for item in response_json["Paintings"]:
                image_links.append(item["image"])
        return image_links

    def preprocess_image(self, image_url, size):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        image = image.resize((size, size), Image.ANTIALIAS)
        image = np.asarray(image)
        image = (image - 127.5) / 127.5
        return image

    def load_dataset(self, artist_name, max_pages=10):
        images = []
        for page in range(1, max_pages + 1):
            print(f"Retrieving images from {artist_name} - Page {page}")
            image_links = self.get_image_links(artist_name, page)
            if not image_links:
                break
            for image_url in image_links:
                try:
                    print(f"Processing image: {image_url}")
                    image = self.preprocess_image(image_url, self.IMAGE_SIZE)
                    images.append(image)
                except:
                    print(f"Error processing image: {image_url}")
                    pass
        return np.array(images)

    def split_datasets(self, bela_dataset, camille_dataset):
        bela_train, bela_test = train_test_split(bela_dataset, test_size=self.TEST_SPLIT, random_state=42)
        camille_train, cammile_test = train_test_split(camille_dataset, test_size=self.TEST_SPLIT, random_state=42)
        print("bela - Train:", bela_train.shape, "Test:", bela_test.shape)
        print("camille - Train:", camille_train.shape, "Test:", cammile_test.shape)
        return bela_train, bela_test, camille_train, cammile_test

