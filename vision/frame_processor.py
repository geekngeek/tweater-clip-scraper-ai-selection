"""
Frame processing utilities for video analysis.
"""

import asyncio
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Process video frames for vision analysis."""
    
    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        quality: int = 85,
        max_frame_size_kb: int = 500,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality
        self.max_frame_size_kb = max_frame_size_kb
    
    async def extract_keyframes(
        self,
        video_path: str,
        num_frames: int = 10,
        method: str = "uniform",
    ) -> List[Dict[str, Any]]:
        """
        Extract keyframes from video for analysis.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            method: Extraction method ('uniform', 'scene_change', 'quality')
            
        Returns:
            List of frame data dictionaries
        """
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video: {duration:.1f}s, {frame_count} frames, {fps:.1f} FPS")
            
            if method == "uniform":
                frames = await self._extract_uniform_frames(cap, num_frames, duration)
            elif method == "scene_change":
                frames = await self._extract_scene_change_frames(cap, num_frames, duration)
            elif method == "quality":
                frames = await self._extract_quality_frames(cap, num_frames, duration)
            else:
                logger.warning(f"Unknown extraction method: {method}, using uniform")
                frames = await self._extract_uniform_frames(cap, num_frames, duration)
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames using {method} method")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    async def _extract_uniform_frames(
        self,
        cap: cv2.VideoCapture,
        num_frames: int,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Extract frames at uniform intervals."""
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= num_frames:
            # Extract every frame if video is short
            frame_indices = list(range(frame_count))
        else:
            # Extract at uniform intervals
            step = frame_count // num_frames
            frame_indices = [i * step for i in range(num_frames)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                timestamp = frame_idx / cap.get(cv2.CAP_PROP_FPS)
                
                frame_data = await self._process_frame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                )
                
                if frame_data:
                    frames.append(frame_data)
        
        return frames
    
    async def _extract_scene_change_frames(
        self,
        cap: cv2.VideoCapture,
        num_frames: int,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Extract frames at scene change boundaries."""
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample every Nth frame for scene change detection
        sample_rate = max(1, frame_count // (num_frames * 3))
        
        prev_hist = None
        scene_changes = []
        
        for frame_idx in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate histogram for scene change detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            if prev_hist is not None:
                # Compare histograms using correlation
                correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                
                # If correlation is low, it's likely a scene change
                if correlation < 0.8:
                    timestamp = frame_idx / fps
                    scene_changes.append((frame_idx, timestamp))
            
            prev_hist = hist
        
        # Select top scene changes
        scene_changes.sort(key=lambda x: x[1])  # Sort by timestamp
        
        # If we have more scene changes than needed, sample uniformly
        if len(scene_changes) > num_frames:
            step = len(scene_changes) // num_frames
            selected_changes = [scene_changes[i * step] for i in range(num_frames)]
        else:
            selected_changes = scene_changes
        
        # Extract frames at scene changes
        for frame_idx, timestamp in selected_changes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_data = await self._process_frame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_index=frame_idx,
                )
                
                if frame_data:
                    frames.append(frame_data)
        
        return frames
    
    async def _extract_quality_frames(
        self,
        cap: cv2.VideoCapture,
        num_frames: int,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """Extract frames with best quality (sharpness, brightness)."""
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample more frames than needed, then select best quality
        sample_count = min(num_frames * 3, frame_count)
        sample_step = frame_count // sample_count
        
        frame_qualities = []
        
        for i in range(sample_count):
            frame_idx = i * sample_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate quality metrics
            quality_score = self._calculate_frame_quality(frame)
            timestamp = frame_idx / fps
            
            frame_qualities.append((frame_idx, timestamp, quality_score, frame))
        
        # Sort by quality and select top frames
        frame_qualities.sort(key=lambda x: x[2], reverse=True)
        selected_frames = frame_qualities[:num_frames]
        
        # Sort by timestamp for chronological order
        selected_frames.sort(key=lambda x: x[1])
        
        frames = []
        for frame_idx, timestamp, quality_score, frame in selected_frames:
            frame_data = await self._process_frame(
                frame=frame,
                timestamp=timestamp,
                frame_index=frame_idx,
                quality_score=quality_score,
            )
            
            if frame_data:
                frames.append(frame_data)
        
        return frames
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score based on sharpness and brightness."""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance for sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Mean brightness
        brightness = gray.mean()
        
        # Contrast (standard deviation)
        contrast = gray.std()
        
        # Combined quality score (weighted)
        quality_score = (
            laplacian_var * 0.5 +  # Sharpness (most important)
            (brightness / 255.0) * 0.3 +  # Normalized brightness
            (contrast / 255.0) * 0.2  # Normalized contrast
        )
        
        return quality_score
    
    async def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        quality_score: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process individual frame for analysis."""
        
        try:
            # Resize frame if needed
            height, width = frame.shape[:2]
            
            if width > self.target_width or height > self.target_height:
                # Calculate scaling factor maintaining aspect ratio
                scale_w = self.target_width / width
                scale_h = self.target_height / height
                scale = min(scale_w, scale_h)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Encode as base64 for API transmission
            base64_image = self._encode_image_to_base64(pil_image)
            
            # Check size
            image_size_kb = len(base64_image) / 1024
            
            if image_size_kb > self.max_frame_size_kb:
                # Reduce quality and try again
                quality = max(50, int(self.quality * 0.7))
                base64_image = self._encode_image_to_base64(pil_image, quality=quality)
                image_size_kb = len(base64_image) / 1024
                
                if image_size_kb > self.max_frame_size_kb:
                    logger.warning(f"Frame at {timestamp:.1f}s too large: {image_size_kb:.1f}KB")
                    return None
            
            return {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "base64_image": base64_image,
                "width": pil_image.width,
                "height": pil_image.height,
                "size_kb": image_size_kb,
                "quality_score": quality_score,
            }
            
        except Exception as e:
            logger.error(f"Frame processing failed at {timestamp:.1f}s: {e}")
            return None
    
    def _encode_image_to_base64(self, image: Image.Image, quality: Optional[int] = None) -> str:
        """Encode PIL Image to base64 string."""
        
        if quality is None:
            quality = self.quality
        
        buffer = BytesIO()
        
        # Convert to RGB if needed (remove alpha channel)
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG with specified quality
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
    
    async def create_video_thumbnail(
        self,
        video_path: str,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Create a thumbnail from video at specified timestamp."""
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Use middle of video if no timestamp specified
            if timestamp is None:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                timestamp = (frame_count / fps) / 2 if fps > 0 else 0
            
            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Process frame
            frame_data = await self._process_frame(
                frame=frame,
                timestamp=timestamp,
                frame_index=0,
            )
            
            return frame_data['base64_image'] if frame_data else None
            
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            return None
    
    def analyze_frame_content(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for basic content characteristics."""
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Motion/blur detection using Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < 100  # Threshold for blur detection
            
            # Brightness analysis
            brightness = gray.mean()
            is_dark = brightness < 50
            is_bright = brightness > 200
            
            # Contrast analysis
            contrast = gray.std()
            is_low_contrast = contrast < 30
            
            return {
                "num_faces": len(faces),
                "has_faces": len(faces) > 0,
                "is_blurry": is_blurry,
                "is_dark": is_dark,
                "is_bright": is_bright,
                "is_low_contrast": is_low_contrast,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "sharpness": float(laplacian_var),
                "face_locations": faces.tolist() if len(faces) > 0 else [],
            }
            
        except Exception as e:
            logger.error(f"Frame content analysis failed: {e}")
            return {
                "num_faces": 0,
                "has_faces": False,
                "is_blurry": True,
                "is_dark": False,
                "is_bright": False,
                "is_low_contrast": True,
                "brightness": 0.0,
                "contrast": 0.0,
                "sharpness": 0.0,
                "face_locations": [],
            }