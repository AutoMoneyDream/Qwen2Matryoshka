"""
Data loading and preprocessing module for multimodal search intent alignment.
Handles both text and video modalities with Qwen2.5-VL preprocessing.
"""

import json
import torch
import random
from PIL import Image
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from .config import get_config


class MultimodalDataset(Dataset):
    """
    Dataset class for handling multimodal query-target pairs.
    Supports four combinations: text-text, text-video, video-text, video-video.
    """
    
    def __init__(self, data_path: str, video_meta_path: str, config=None, split='train'):
        """
        Initialize the multimodal dataset.
        
        Args:
            data_path: Path to the query-target pairs data (JSONL format)
            video_meta_path: Path to the video metadata (JSONL format)
            config: Configuration object
            split: Dataset split ('train', 'eval', 'test')
        """
        self.config = config or get_config()
        self.split = split
        
        # Load query-target pairs
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Warning: Data file {data_path} not found. Creating empty dataset.")
            self.data = []
        
        # Load video metadata and create lookup dictionary
        self.video_meta = {}
        try:
            with open(video_meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    meta = json.loads(line.strip())
                    video_id = meta.get('video_id', meta.get('id'))
                    if video_id:
                        self.video_meta[video_id] = meta
        except FileNotFoundError:
            print(f"Warning: Video metadata file {video_meta_path} not found. Creating empty metadata.")
            self.video_meta = {}
    
    def __len__(self) -> int:
        """Return the number of query-target pairs."""
        return len(self.data)
    
    def _process_video_content(self, video_id: str) -> tuple[str, Optional[Image.Image]]:
        """
        Process video content by extracting text and image information.
        
        Args:
            video_id: The video identifier
            
        Returns:
            Tuple of (concatenated_text, representative_image)
        """
        if video_id not in self.video_meta:
            return f"Video ID: {video_id}", None
        
        meta = self.video_meta[video_id]
        
        # Concatenate all text fields
        text_fields = []
        for field in ['caption', 'title', 'text', 'ocr', 'asr']:
            if field in meta and meta[field]:
                if isinstance(meta[field], list):
                    text_fields.extend(str(item) for item in meta[field])
                else:
                    text_fields.append(str(meta[field]))
        
        # Combine all text with newlines
        combined_text = '\n'.join(text_fields) if text_fields else f"Video ID: {video_id}"
        
        # Get representative image
        image = None
        if 'images' in meta and meta['images']:
            try:
                # Use first image or random selection
                image_path = meta['images'][0] if len(meta['images']) == 1 else random.choice(meta['images'])
                if isinstance(image_path, str):
                    image = Image.open(image_path).convert('RGB')
                elif isinstance(image_path, dict) and 'path' in image_path:
                    image = Image.open(image_path['path']).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image for video {video_id}: {e}")
                image = None
        
        return combined_text, image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single query-target pair.
        
        Args:
            idx: Index of the data item
            
        Returns:
            Dictionary containing query and target information
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        item = self.data[idx]
        query = item.get('query', '')
        target = item.get('target', '')
        
        # Initialize return dictionary
        result = {
            'query_text': '',
            'query_image': None,
            'target_text': '',
            'target_image': None,
            'raw_query': query,
            'raw_target': target
        }
        
        # Process query
        if self._is_video_id(query):
            result['query_text'], result['query_image'] = self._process_video_content(query)
        else:
            result['query_text'] = str(query)
            result['query_image'] = None
        
        # Process target
        if self._is_video_id(target):
            result['target_text'], result['target_image'] = self._process_video_content(target)
        else:
            result['target_text'] = str(target)
            result['target_image'] = None
        
        return result
    
    def _is_video_id(self, content: Any) -> bool:
        """
        Determine if content is a video ID.
        
        Args:
            content: Content to check
            
        Returns:
            True if content appears to be a video ID
        """
        if not isinstance(content, str):
            return False
        
        # Check if it's in video metadata
        if content in self.video_meta:
            return True
        
        # Heuristic: check if it looks like a video ID
        # (could be customized based on your video ID format)
        if len(content) < 200 and ('video' in content.lower() or content.isalnum()):
            return True
        
        return False


def collate_fn(batch: List[Dict[str, Any]], config) -> Dict[str, Any]:
    """
    Custom collate function for batching multimodal data.
    
    Args:
        batch: List of data items from __getitem__
        config: Configuration object containing processor
        
    Returns:
        Batched and processed data ready for model input
    """
    # Load processor from config
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code
    )
    
    # Separate query and target data
    query_texts = []
    query_images = []
    target_texts = []
    target_images = []
    
    for item in batch:
        query_texts.append(item['query_text'])
        query_images.append(item['query_image'])
        target_texts.append(item['target_text'])
        target_images.append(item['target_image'])
    
    # Process queries
    query_messages = []
    for i, (text, image) in enumerate(zip(query_texts, query_images)):
        message = {
            "role": "user",
            "content": []
        }
        
        if image is not None:
            message["content"].append({
                "type": "image",
                "image": image
            })
        
        message["content"].append({
            "type": "text", 
            "text": text[:config.max_text_length]  # Truncate if too long
        })
        
        query_messages.append([message])
    
    # Process targets
    target_messages = []
    for i, (text, image) in enumerate(zip(target_texts, target_images)):
        message = {
            "role": "user",
            "content": []
        }
        
        if image is not None:
            message["content"].append({
                "type": "image",
                "image": image
            })
        
        message["content"].append({
            "type": "text",
            "text": text[:config.max_text_length]  # Truncate if too long
        })
        
        target_messages.append([message])
    
    # Use processor to encode queries and targets
    try:
        query_inputs = processor.apply_chat_template(
            query_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_length
        )
        
        target_inputs = processor.apply_chat_template(
            target_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_length
        )
    except Exception as e:
        print(f"Warning: Error in processor.apply_chat_template: {e}")
        # Fallback to simple text processing
        query_inputs = processor(
            text=query_texts,
            images=[img for img in query_images if img is not None] or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_length
        )
        
        target_inputs = processor(
            text=target_texts,
            images=[img for img in target_images if img is not None] or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_length
        )
    
    return {
        'query_inputs': query_inputs,
        'target_inputs': target_inputs,
        'batch_size': len(batch)
    }


def create_dataloader(
    dataset: MultimodalDataset,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    distributed: bool = False
) -> DataLoader:
    """
    Create a DataLoader for multimodal data.
    
    Args:
        dataset: MultimodalDataset instance
        batch_size: Batch size (uses config default if None)
        shuffle: Whether to shuffle data (uses config default if None)
        num_workers: Number of worker processes (uses config default if None)
        distributed: Whether to use distributed training
        
    Returns:
        Configured DataLoader instance
    """
    config = dataset.config
    
    # Use config defaults if not specified
    if batch_size is None:
        batch_size = config.batch_size
    if shuffle is None:
        shuffle = config.shuffle
    if num_workers is None:
        num_workers = config.num_workers
    
    # Setup distributed sampler if needed
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler handles shuffling in distributed mode
    
    def collate_wrapper(batch):
        return collate_fn(batch, dataset.config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_wrapper,
        drop_last=True  # Ensure consistent batch sizes for training
    )
    
    return dataloader


def create_sample_data(data_path: str, video_meta_path: str, num_samples: int = 10):
    """
    Create sample data files for testing purposes.
    
    Args:
        data_path: Path where to create sample query-target data
        video_meta_path: Path where to create sample video metadata
        num_samples: Number of sample pairs to create
    """
    import os
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_meta_path), exist_ok=True)
    
    # Create sample query-target pairs
    sample_data = []
    for i in range(num_samples):
        if i % 4 == 0:  # text-text
            sample_data.append({
                "query": f"Sample query text {i}",
                "target": f"Sample target text {i}"
            })
        elif i % 4 == 1:  # text-video
            sample_data.append({
                "query": f"Sample query text {i}",
                "target": f"video_{i:03d}"
            })
        elif i % 4 == 2:  # video-text
            sample_data.append({
                "query": f"video_{i:03d}",
                "target": f"Sample target text {i}"
            })
        else:  # video-video
            sample_data.append({
                "query": f"video_{i:03d}",
                "target": f"video_{(i+1):03d}"
            })
    
    # Save query-target pairs
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create sample video metadata
    sample_videos = []
    for i in range(num_samples):
        video_meta = {
            "video_id": f"video_{i:03d}",
            "caption": f"This is a sample video {i} with interesting content",
            "title": f"Sample Video {i}",
            "text": f"Additional text description for video {i}",
            "ocr": f"OCR text from video {i}",
            "asr": f"Audio transcript from video {i}",
            "images": []  # Empty for sample - would contain actual image paths
        }
        sample_videos.append(video_meta)
    
    # Save video metadata
    with open(video_meta_path, 'w', encoding='utf-8') as f:
        for video in sample_videos:
            f.write(json.dumps(video, ensure_ascii=False) + '\n')
    
    print(f"Created {num_samples} sample data pairs at {data_path}")
    print(f"Created {num_samples} sample video metadata at {video_meta_path}")


if __name__ == "__main__":
    # Test the data loader
    from transformers import Qwen2VLProcessor
    from .config import get_config
    
    config = get_config()
    
    # Create sample data for testing
    create_sample_data(config.data_path, config.video_meta_path)
    
    # Load processor
    try:
        processor = Qwen2VLProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            config.data_path,
            config.video_meta_path,
            processor
        )
        
        # Test first batch
        first_batch = next(iter(dataloader))
        print("Successfully created dataloader!")
        print(f"Batch size: {first_batch['batch_size']}")
        print(f"Query input keys: {first_batch['query_input'].keys()}")
        print(f"Target input keys: {first_batch['target_input'].keys()}")
        
        # Print shapes
        for key, value in first_batch['query_input'].items():
            if isinstance(value, torch.Tensor):
                print(f"Query {key} shape: {value.shape}")
        
        for key, value in first_batch['target_input'].items():
            if isinstance(value, torch.Tensor):
                print(f"Target {key} shape: {value.shape}")
                
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        print("This is expected if Qwen2.5-VL model is not available locally.")