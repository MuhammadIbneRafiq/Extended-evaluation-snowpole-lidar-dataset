"""
Merkle Tree Implementation for Inference Caching
Efficient tile-based caching system for accelerating inference
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import pickle
from collections import OrderedDict
import time


class TileHash:
    """Compute hash for image tiles"""
    
    def __init__(self, algorithm: str = 'sha256'):
        self.algorithm = algorithm
    
    def compute_hash(self, tile: torch.Tensor) -> str:
        """
        Compute hash for a tile
        
        Args:
            tile: Tensor representing an image tile
        
        Returns:
            Hash string
        """
        # Convert tile to bytes
        tile_bytes = tile.cpu().numpy().tobytes()
        
        # Compute hash
        if self.algorithm == 'sha256':
            hash_obj = hashlib.sha256(tile_bytes)
        elif self.algorithm == 'md5':
            hash_obj = hashlib.md5(tile_bytes)
        else:
            raise ValueError(f"Unknown hash algorithm: {self.algorithm}")
        
        return hash_obj.hexdigest()
    
    def compute_perceptual_hash(self, tile: torch.Tensor, 
                               threshold: float = 0.95) -> Tuple[str, float]:
        """
        Compute perceptual hash with similarity score
        
        Args:
            tile: Tensor representing an image tile
            threshold: Similarity threshold
        
        Returns:
            Tuple of (hash, similarity_score)
        """
        # Downsample tile for perceptual hashing
        if tile.dim() == 4:
            tile = tile.squeeze(0)
        if tile.dim() == 3:
            # Average across channels
            tile = tile.mean(dim=0)
        
        # Resize to 8x8 for perceptual hash
        tile_small = nn.functional.interpolate(
            tile.unsqueeze(0).unsqueeze(0),
            size=(8, 8),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Compute DCT-like transform
        mean_val = tile_small.mean()
        binary_hash = (tile_small > mean_val).float()
        
        # Convert to string hash
        hash_str = ''.join([str(int(b)) for b in binary_hash.flatten().tolist()])
        
        # Compute similarity score (for comparison with cached tiles)
        similarity = 1.0  # Default for new tiles
        
        return hash_str, similarity


class MerkleNode:
    """Node in the Merkle tree"""
    
    def __init__(self, hash_value: str = None, data: Any = None,
                 left: 'MerkleNode' = None, right: 'MerkleNode' = None):
        self.hash_value = hash_value
        self.data = data  # Stores detection results for leaf nodes
        self.left = left
        self.right = right
        self.is_leaf = (left is None and right is None)
    
    def compute_hash(self) -> str:
        """Compute hash for internal nodes"""
        if self.is_leaf:
            return self.hash_value
        else:
            combined = (self.left.hash_value or '') + (self.right.hash_value or '')
            return hashlib.sha256(combined.encode()).hexdigest()


class MerkleTree:
    """
    Merkle Tree for caching inference results
    Enables efficient detection of changed regions
    """
    
    def __init__(self, tile_size: int = 128, cache_size: int = 1000,
                 similarity_threshold: float = 0.95):
        self.tile_size = tile_size
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        # Cache for storing detection results
        self.cache = OrderedDict()
        self.tile_hasher = TileHash()
        
        # Merkle tree structure
        self.root = None
        self.leaf_nodes = {}
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tiles': 0,
            'cached_tiles': 0
        }
    
    def split_into_tiles(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Split image into tiles
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            List of tile dictionaries with position and tensor
        """
        B, C, H, W = image.shape
        tiles = []
        
        # Calculate number of tiles
        num_tiles_h = (H + self.tile_size - 1) // self.tile_size
        num_tiles_w = (W + self.tile_size - 1) // self.tile_size
        
        for b in range(B):
            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    # Extract tile
                    start_h = i * self.tile_size
                    end_h = min(start_h + self.tile_size, H)
                    start_w = j * self.tile_size
                    end_w = min(start_w + self.tile_size, W)
                    
                    tile = image[b:b+1, :, start_h:end_h, start_w:end_w]
                    
                    # Pad if necessary
                    if tile.shape[-2] < self.tile_size or tile.shape[-1] < self.tile_size:
                        pad_h = self.tile_size - tile.shape[-2]
                        pad_w = self.tile_size - tile.shape[-1]
                        tile = nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    
                    tiles.append({
                        'batch_idx': b,
                        'row': i,
                        'col': j,
                        'position': (start_h, end_h, start_w, end_w),
                        'tensor': tile
                    })
        
        self.stats['total_tiles'] = len(tiles)
        return tiles
    
    def build_tree(self, tiles: List[Dict[str, Any]]) -> MerkleNode:
        """
        Build Merkle tree from tiles
        
        Args:
            tiles: List of tile dictionaries
        
        Returns:
            Root node of the Merkle tree
        """
        # Create leaf nodes
        leaf_nodes = []
        for tile_info in tiles:
            tile_hash = self.tile_hasher.compute_hash(tile_info['tensor'])
            node = MerkleNode(hash_value=tile_hash, data=tile_info)
            leaf_nodes.append(node)
            
            # Store in leaf node dictionary for quick access
            key = f"{tile_info['batch_idx']}_{tile_info['row']}_{tile_info['col']}"
            self.leaf_nodes[key] = node
        
        # Build tree bottom-up
        current_level = leaf_nodes
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else None
                
                if right is None:
                    # Odd number of nodes, promote the last one
                    next_level.append(left)
                else:
                    # Create internal node
                    internal_node = MerkleNode(left=left, right=right)
                    internal_node.hash_value = internal_node.compute_hash()
                    next_level.append(internal_node)
            
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
        return self.root
    
    def find_changed_tiles(self, new_image: torch.Tensor, 
                          prev_tree: Optional['MerkleTree'] = None) -> List[Dict[str, Any]]:
        """
        Find tiles that have changed compared to previous frame
        
        Args:
            new_image: New image tensor
            prev_tree: Previous Merkle tree (if available)
        
        Returns:
            List of changed tile dictionaries
        """
        tiles = self.split_into_tiles(new_image)
        changed_tiles = []
        
        for tile_info in tiles:
            tile_hash = self.tile_hasher.compute_hash(tile_info['tensor'])
            key = f"{tile_info['batch_idx']}_{tile_info['row']}_{tile_info['col']}"
            
            # Check if tile exists in cache
            if tile_hash in self.cache:
                self.stats['cache_hits'] += 1
                tile_info['cached'] = True
                tile_info['detection_result'] = self.cache[tile_hash]
            else:
                self.stats['cache_misses'] += 1
                tile_info['cached'] = False
                changed_tiles.append(tile_info)
                
                # Check if similar tile exists (perceptual hashing)
                if self.similarity_threshold < 1.0:
                    perceptual_hash, _ = self.tile_hasher.compute_perceptual_hash(
                        tile_info['tensor']
                    )
                    
                    # Look for similar tiles in cache
                    for cached_hash, cached_result in self.cache.items():
                        if self._compute_similarity(tile_hash, cached_hash) > self.similarity_threshold:
                            tile_info['cached'] = True
                            tile_info['detection_result'] = cached_result
                            changed_tiles.remove(tile_info)
                            break
        
        self.stats['cached_tiles'] = len(tiles) - len(changed_tiles)
        return changed_tiles
    
    def update_cache(self, tile_info: Dict[str, Any], detection_result: Any):
        """
        Update cache with detection results for a tile
        
        Args:
            tile_info: Tile information dictionary
            detection_result: Detection results for the tile
        """
        tile_hash = self.tile_hasher.compute_hash(tile_info['tensor'])
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.popitem(last=False)
        
        self.cache[tile_hash] = detection_result
    
    def merge_tile_results(self, tiles: List[Dict[str, Any]], 
                          image_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Merge detection results from tiles back into full image
        
        Args:
            tiles: List of tile dictionaries with detection results
            image_shape: Original image shape
        
        Returns:
            Merged detection results
        """
        B, C, H, W = image_shape
        merged_results = []
        
        for b in range(B):
            # Create empty result tensor for this batch
            batch_result = torch.zeros((C, H, W), dtype=torch.float32)
            confidence_map = torch.zeros((H, W), dtype=torch.float32)
            
            # Merge tiles for this batch
            batch_tiles = [t for t in tiles if t['batch_idx'] == b]
            
            for tile_info in batch_tiles:
                if 'detection_result' in tile_info:
                    start_h, end_h, start_w, end_w = tile_info['position']
                    
                    # Extract relevant portion of detection result
                    result = tile_info['detection_result']
                    if isinstance(result, torch.Tensor):
                        # Crop to actual tile size (removing padding)
                        actual_h = end_h - start_h
                        actual_w = end_w - start_w
                        result_cropped = result[:, :actual_h, :actual_w]
                        
                        # Place in merged result
                        batch_result[:, start_h:end_h, start_w:end_w] = result_cropped
                        confidence_map[start_h:end_h, start_w:end_w] = 1.0
            
            merged_results.append(batch_result)
        
        return torch.stack(merged_results)
    
    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two hashes
        
        Args:
            hash1: First hash string
            hash2: Second hash string
        
        Returns:
            Similarity score [0, 1]
        """
        # Simple Hamming distance for binary hashes
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get caching statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / max(total_requests, 1)
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'total_tiles': self.stats['total_tiles'],
            'cached_tiles': self.stats['cached_tiles'],
            'cache_size': len(self.cache),
            'efficiency': self.stats['cached_tiles'] / max(self.stats['total_tiles'], 1)
        }


class InferenceCache:
    """
    High-level inference caching system using Merkle trees
    """
    
    def __init__(self, model: nn.Module, tile_size: int = 128,
                 cache_size: int = 1000, similarity_threshold: float = 0.95):
        self.model = model
        self.merkle_tree = MerkleTree(tile_size, cache_size, similarity_threshold)
        self.tile_size = tile_size
        
        # Timing statistics
        self.timing_stats = {
            'tile_processing': 0.0,
            'inference': 0.0,
            'merging': 0.0,
            'total': 0.0
        }
    
    @torch.no_grad()
    def cached_inference(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform inference with caching
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            Tuple of (detection results, statistics)
        """
        start_time = time.time()
        
        # Split image into tiles
        tile_start = time.time()
        tiles = self.merkle_tree.split_into_tiles(image)
        changed_tiles = self.merkle_tree.find_changed_tiles(image)
        self.timing_stats['tile_processing'] = time.time() - tile_start
        
        # Process only changed tiles
        inference_start = time.time()
        if changed_tiles:
            # Stack changed tiles for batch processing
            tile_batch = torch.cat([t['tensor'] for t in changed_tiles], dim=0)
            
            # Run inference on changed tiles
            self.model.eval()
            tile_results = self.model(tile_batch)
            
            # Unpack results and update cache
            for i, tile_info in enumerate(changed_tiles):
                if isinstance(tile_results, list):
                    # Handle multi-scale outputs
                    result = [r[i:i+1] for r in tile_results]
                else:
                    result = tile_results[i:i+1]
                
                tile_info['detection_result'] = result
                self.merkle_tree.update_cache(tile_info, result)
        
        self.timing_stats['inference'] = time.time() - inference_start
        
        # Merge results
        merge_start = time.time()
        
        # Combine cached and new results
        all_tiles = tiles
        for tile in all_tiles:
            if tile.get('cached', False) and 'detection_result' not in tile:
                # Retrieve from cache
                tile_hash = self.merkle_tree.tile_hasher.compute_hash(tile['tensor'])
                tile['detection_result'] = self.merkle_tree.cache.get(tile_hash)
        
        # Create full detection output
        # Note: This is simplified - actual merging would need proper NMS
        detection_results = self._merge_detections(all_tiles, image.shape)
        
        self.timing_stats['merging'] = time.time() - merge_start
        self.timing_stats['total'] = time.time() - start_time
        
        # Compute speedup
        stats = self.merkle_tree.get_statistics()
        stats['timing'] = self.timing_stats.copy()
        stats['speedup'] = 1.0 / max(1.0 - stats['efficiency'], 0.1)  # Theoretical speedup
        
        return detection_results, stats
    
    def _merge_detections(self, tiles: List[Dict[str, Any]], 
                         image_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """
        Merge detection results from tiles
        
        Args:
            tiles: List of tiles with detection results
            image_shape: Original image shape
        
        Returns:
            Merged detection results
        """
        # This is a simplified merge - actual implementation would need
        # proper coordinate transformation and NMS
        
        all_detections = []
        
        for tile in tiles:
            if 'detection_result' in tile:
                result = tile['detection_result']
                position = tile['position']
                
                # Adjust detection coordinates based on tile position
                # This would need actual coordinate transformation
                all_detections.append(result)
        
        # Return first valid detection for now (simplified)
        return all_detections[0] if all_detections else None
    
    def reset_cache(self):
        """Reset the cache"""
        self.merkle_tree.cache.clear()
        self.merkle_tree.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tiles': 0,
            'cached_tiles': 0
        }


if __name__ == "__main__":
    # Test Merkle tree caching
    print("Testing Merkle Tree Inference Caching...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            # Simulate detection output
            return torch.randn_like(x)
    
    model = DummyModel()
    
    # Create inference cache
    cache = InferenceCache(model, tile_size=128, cache_size=100)
    
    # Test with dummy images
    batch_size = 1
    image1 = torch.randn(batch_size, 3, 640, 640)
    
    # First inference (no cache)
    print("\nFirst inference (no cache):")
    results1, stats1 = cache.cached_inference(image1)
    print(f"Statistics: {stats1}")
    
    # Second inference with same image (should use cache)
    print("\nSecond inference (with cache):")
    results2, stats2 = cache.cached_inference(image1)
    print(f"Statistics: {stats2}")
    
    # Third inference with slightly modified image
    image2 = image1.clone()
    image2[:, :, 100:200, 100:200] += 0.5  # Modify a region
    
    print("\nThird inference (partial cache):")
    results3, stats3 = cache.cached_inference(image2)
    print(f"Statistics: {stats3}")
    
    print("\nMerkle tree caching test completed!")
