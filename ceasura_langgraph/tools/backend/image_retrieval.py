from contextlib import ExitStack, contextmanager
from PIL import Image
from transformers import AutoProcessor, BlipForImageTextRetrieval
import chromadb
from chromadb.config import Settings
import torch
from tqdm import tqdm
from torch.nn.functional import normalize
import uuid
from pathlib import Path
import re
import logging



import numpy as np
import logging
from PIL import ImageFile
import concurrent


logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
CHROMADB_PATH = Path(".chromadb/")
IMAGE_PATH = Path(".images/")
IMAGE_MAX_PIXELS = 20_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_paths_from_images(images):
    """Returns the paths of the images."""
    # images = array(["<IMAGE stored at 'datasets/art/images/img_13.jpg'>", ...)
    paths = [re.search(r"'(.*)'", x).group(1) for x in images if x is not None]
    return paths


class ImageRetriever():
    def __init__(self, init_db=True):
        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.index = dict()
        self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(CHROMADB_PATH),
            ))

    def retrieve(self, image_paths: str, query: str, table_name: str, column: str, threshold=1.0, batch_size=10):
        """Retrieves images from the database.

        Args:
            image_paths (list): list of image paths.
            query (str): text query.
            threshold (float): threshold for similarity score.
        Returns:
            list: list of image paths that are similar to the query.   # TODO separate index per table
        """
        text_embedding = self.get_text_embeddings(query)[0].tolist()
        result = self.index[column].query(
            query_embeddings=text_embedding,
            n_results=min(100, self.index[column].count()),
        )
        downsized_paths = result["documents"][0]
        image_paths = result["ids"][0]

        result = []
        for i in range(0, len(image_paths), batch_size):
            with ExitStack() as stack:
                images = [stack.enter_context(Image.open(p)) for p in downsized_paths[i: i + batch_size]]
                inputs = self.processor(images=images, text=query, return_tensors="pt")
                outputs = self.model(**inputs, use_itm_head=True)

                distance = outputs.itm_score[:, 0].view(-1)
                # sort images by distance
                distance, indices = distance.view(-1).sort(dim=-1, descending=False)
                result += [image_paths[i + j] for j in indices.tolist() if distance[j] < threshold]
        return result

    

    @contextmanager
    def load_images(self, image_paths):
        """Loads images from the database. If the image is too large, it is down-sampled."""
        IMAGE_PATH.mkdir(exist_ok=True)
        result_images = []
        result_paths = []
        with ExitStack() as stack:
            for path in image_paths:
                image = stack.enter_context(Image.open(path))
                if np.prod(image.size) > IMAGE_MAX_PIXELS:
                    ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                    image.thumbnail((int(image.size[0] * ratio), int(image.size[1] * ratio)))
                # store image in cache directory
                name = uuid.uuid4().hex + ".png"
                try:
                    image.convert('RGB').save(IMAGE_PATH / name)
                    result_paths.append(str(IMAGE_PATH / name))
                    result_images.append(image)
                except Exception as e:
                    logger.warning(str(e))
                    pass
            yield result_images, result_paths

    def get_visual_embeddings(self, image_paths, table_name, column):
        """Return embeddings for images.
        
        Args:
            image_paths (list): list of image paths
            table_name (str): name of table
            column (str): name of column
        Returns:
            embeddings for images and metadata to be stored in chromadb.
        """
        with self.load_images(image_paths) as (batch_images, downsized_paths):
            inputs = self.processor(images=batch_images, return_tensors="pt")
            outputs = self.model.vision_model(**inputs)[0]
            image_feat = normalize(self.model.vision_proj(outputs[:, 0, :]), dim=-1)
            return dict(
                embeddings=image_feat.tolist(),
                documents=downsized_paths,
                metadatas=[{"table": table_name, "column": column} for _ in range(len(image_paths))],
                ids=image_paths
            )

    def get_text_embeddings(self, query):
        """Return embeddings for text.
        
        Args:
            query (str): text query
        Returns:
            embeddings for text
        """
        inputs = self.processor(text=query, return_tensors="pt")
        question_embeds = self.model.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=None,
            return_dict=False,
        )
        text_feat = normalize(self.model.text_proj(question_embeds[0][:, 0, :]), dim=-1)
        return text_feat


    def persist(self):
        """Persist the index."""
        if self.client is not None:
            self.client.persist()
