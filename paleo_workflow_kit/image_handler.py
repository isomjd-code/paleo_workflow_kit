# paleo_workflow_kit/image_handler.py

import requests
from PIL import Image, ImageDraw, ImageFont
import os
import io
import logging
import math
import base64
import mimetypes
from typing import List, Dict, Any, Optional, Tuple

# Get logger for this module
logger = logging.getLogger(__name__)

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

class ImageHandler:
    """
    Handles image downloading, processing (drawing, rotation, segmentation),
    and encoding tasks.
    """

    def __init__(self,
                 default_font_path: Optional[str] = "DejaVuSans.ttf",
                 fallback_font_path: Optional[str] = None,
                 default_font_size: int = 20):
        """
        Initializes the ImageHandler.

        Args:
            default_font_path: Default path to a TTF/OTF font file for drawing.
            fallback_font_path: Optional path to a fallback font file.
            default_font_size: Default font size for drawing text.
        """
        self.default_font_path = default_font_path
        self.fallback_font_path = fallback_font_path
        self.default_font_size = default_font_size
        logger.info("ImageHandler initialized.")

    def _load_font(self, font_path: Optional[str], fallback_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        """Loads a font with fallback options."""
        font = None
        primary_path = font_path or self.default_font_path
        fallback = fallback_path or self.fallback_font_path

        if primary_path:
            try:
                font = ImageFont.truetype(primary_path, size)
                logger.debug(f"Loaded primary font: {primary_path} (size {size})")
                return font
            except IOError:
                logger.warning(f"Primary font not found or failed to load: {primary_path}")
                if fallback:
                    try:
                        font = ImageFont.truetype(fallback, size)
                        logger.info(f"Loaded fallback font: {fallback} (size {size})")
                        return font
                    except IOError:
                        logger.warning(f"Fallback font also failed: {fallback}")

        # If primary and fallback failed, try common system fonts
        common_fonts = ["arial.ttf", "LiberationSans-Regular.ttf"]
        for common in common_fonts:
             try:
                 font = ImageFont.truetype(common, size)
                 logger.info(f"Loaded system font: {common} (size {size})")
                 return font
             except IOError:
                 pass # Try next common font

        # If all else fails, use default PIL font
        logger.warning("No specified, fallback, or common TTF fonts found. Using default PIL font.")
        try:
            # load_default() doesn't take size, adjust later if needed?
            # For now, just load it. Size might be fixed.
            font = ImageFont.load_default()
            return font
        except Exception as e_font:
            logger.error(f"Could not load default PIL font: {e_font}")
            raise ValueError("Failed to load any suitable font.") from e_font

    def download_image(self, url: str) -> Optional[bytes]:
        """Downloads an image from a URL, returning image bytes."""
        logger.info(f"Downloading image from: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; PaleoWorkflowKit/1.0)'}
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            image_bytes = response.content
            # Basic verification that it's image data
            try:
                Image.open(io.BytesIO(image_bytes)).verify()
            except Exception as img_err:
                logger.error(f"Downloaded content from {url} is not valid image data: {img_err}")
                return None
            logger.info(f"Image downloaded successfully ({len(image_bytes)} bytes).")
            return image_bytes
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading image from {url}")
            return None
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if e.response is not None else "N/A"
            logger.error(f"Error downloading image from {url} (Status: {status}): {e}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error downloading image from {url}: {e}", exc_info=True)
             return None

    def _draw_single_number(
        self, draw: ImageDraw.ImageDraw, text: str, base_point: Tuple[int, int],
        font: ImageFont.FreeTypeFont, font_size: int, text_color: Tuple[int, int, int, int],
        text_bg_color: Optional[Tuple[int, int, int, int]], offset_x: int, offset_y: int,
        img_width: int, img_height: int, anchor_side: str = 'left'
    ):
        """Internal helper to draw a single line number with background."""
        # (Logic copied from original script, no changes needed here)
        try:
            if anchor_side == 'left': text_x = base_point[0] - offset_x; text_y = base_point[1] - offset_y - font_size
            elif anchor_side == 'right': text_x = base_point[0] + offset_x; text_y = base_point[1] - offset_y - font_size
            else: text_x = base_point[0] - offset_x; text_y = base_point[1] - offset_y - font_size
            try:
                bbox = draw.textbbox((text_x, text_y), text, font=font, anchor="lt"); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
            except AttributeError:
                if hasattr(font, 'getsize'): text_width, text_height = font.getsize(text); bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
                else: text_width = font_size * len(text) * 0.6; text_height = font_size * 1.2; bbox = (text_x, text_y, text_x + text_width, text_y + text_height); logger.warning("Cannot accurately get text size with default font.")
            final_text_x = max(5, bbox[0])
            if final_text_x + text_width > img_width - 5: final_text_x = img_width - 5 - text_width
            final_text_y = max(5, bbox[1])
            if final_text_y + text_height > img_height - 5: final_text_y = img_height - 5 - text_height
            final_bbox = (final_text_x, final_text_y, final_text_x + text_width, final_text_y + text_height)
            if text_bg_color:
                bg_padding = 3; bg_box = (final_bbox[0] - bg_padding, final_bbox[1] - bg_padding, final_bbox[2] + bg_padding, final_bbox[3] + bg_padding)
                bg_box = (max(0, bg_box[0]), max(0, bg_box[1]), min(img_width, bg_box[2]), min(img_height, bg_box[3]))
                draw.rectangle(bg_box, fill=text_bg_color)
            if hasattr(draw, 'textbbox'): draw.text((final_text_x, final_text_y), text, fill=text_color, font=font, anchor="lt")
            else: draw.text((final_text_x, final_text_y), text, fill=text_color, font=font)
        except Exception as e: logger.error(f"Error drawing number '{text}' at point {base_point}: {e}", exc_info=True)


    def draw_baselines_numbers(
        self, image_bytes: bytes, lines_data: List[Dict[str, Any]], output_path: str,
        line_color: Tuple[int, int, int, int] = (255, 0, 0, 255), line_width: int = 2,
        font_path: Optional[str] = None, fallback_font_path: Optional[str] = None, font_size: Optional[int] = None,
        text_color: Tuple[int, int, int, int] = (255, 0, 0, 255), text_bg_color: Optional[Tuple[int, int, int, int]] = (255, 255, 255, 180),
        text_offset_x: int = 25, text_offset_y: int = 10, output_format: Optional[str] = "JPEG",
        jpeg_quality: int = 95, png_compress_level: int = 1
    ) -> Optional[str]:
        """Draws baselines and line numbers (start & end) on an image and saves it."""
        logger.info(f"Drawing baselines/numbers on image, saving to: {output_path}")
        _font_size = font_size or self.default_font_size
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGBA if needed for drawing transparent elements
            if image.mode not in ('RGB', 'RGBA'): image = image.convert("RGBA")
            elif image.mode == 'RGB' and (any(c[3] < 255 for c in [line_color, text_color]) or (text_bg_color and text_bg_color[3] < 255)):
                image = image.convert("RGBA")

            draw = ImageDraw.Draw(image, image.mode)
            font = self._load_font(font_path, fallback_font_path, _font_size)
            img_width, img_height = image.size

            for i, line in enumerate(lines_data):
                coords = line.get('baseline_coords')
                line_number = i + 1
                line_number_str = str(line_number)
                if coords and len(coords) >= 2:
                    int_coords = [(int(round(p[0])), int(round(p[1]))) for p in coords]
                    draw.line(int_coords, fill=line_color, width=line_width)
                    start_point = int_coords[0]
                    self._draw_single_number(draw, line_number_str, start_point, font, _font_size, text_color, text_bg_color, text_offset_x, text_offset_y, img_width, img_height, 'left')
                    end_point = int_coords[-1]
                    self._draw_single_number(draw, line_number_str, end_point, font, _font_size, text_color, text_bg_color, text_offset_x, text_offset_y, img_width, img_height, 'right')
                else: logger.warning(f"Skipping line {line_number}: Invalid/missing baseline_coords.")

            # --- Saving Logic (copied and adapted) ---
            save_options = {}
            output_path_lower = output_path.lower()
            file_extension = '.' + output_path_lower.split('.')[-1] if '.' in output_path_lower else None
            fmt = output_format or (Image.registered_extensions().get(file_extension) if file_extension else None)
            if fmt: fmt = fmt.upper(); logger.info(f"Saving image as {fmt} to {output_path}")
            else: logger.warning(f"Could not determine save format from extension '{file_extension}'. Saving without explicit format.")

            if fmt == "JPEG":
                save_options['quality'] = jpeg_quality
                image_to_save = image
                if image.mode == 'RGBA':
                    logger.warning("Image has alpha channel (RGBA), converting to RGB for JPEG saving.")
                    bg = Image.new("RGB", image.size, (255, 255, 255))
                    try: bg.paste(image, mask=image.split()[3])
                    except IndexError: bg.paste(image)
                    image_to_save = bg
                elif image.mode != 'RGB': image_to_save = image.convert('RGB')
                image_to_save.save(output_path, "JPEG", **save_options)
            elif fmt == "PNG":
                save_options['compress_level'] = png_compress_level
                image.save(output_path, "PNG", **save_options)
            else:
                try: image.save(output_path, format=output_format, **save_options)
                except KeyError: logger.warning(f"Format '{fmt or output_format}' not recognized. Saving without explicit format."); image.save(output_path, **save_options)
                except OSError as e_save:
                    logger.error(f"Error saving image in format ({fmt}): {e_save}")
                    try:
                        fallback_path = output_path + ".png"; logger.warning(f"Attempting PNG fallback: {fallback_path}")
                        img_fb = image.convert('RGBA') if image.mode == 'P' else image
                        img_fb.save(fallback_path, "PNG", compress_level=png_compress_level); return fallback_path
                    except Exception as e_fallback: logger.error(f"PNG fallback save failed: {e_fallback}"); raise e_save

            logger.info(f"Image with drawings saved successfully to: {output_path}")
            return output_path

        except (FileNotFoundError, IOError, ValueError, ImageProcessingError) as e:
            logger.error(f"Error during baseline/number drawing: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in draw_baselines_numbers: {e}", exc_info=True)
            return None

    def draw_chunk_baselines_numbers(
        self, image_object: Image.Image, lines_data_full_page: List[Dict[str, Any]],
        start_index: int, end_index: int, crop_box: Tuple[int, int, int, int],
        line_color: Tuple[int, int, int, int] = (255, 0, 0, 255), line_width: int = 2,
        font_path: Optional[str] = None, fallback_font_path: Optional[str] = None, font_size: Optional[int] = None,
        text_color: Tuple[int, int, int, int] = (255, 0, 0, 255), text_bg_color: Optional[Tuple[int, int, int, int]] = (255, 255, 255, 180),
        text_offset_x: int = 25, text_offset_y: int = 10
    ) -> Optional[Image.Image]:
        """Draws baselines/numbers for a specific chunk onto a cropped PIL image object."""
        logger.debug(f"Drawing lines/numbers for chunk {start_index+1}-{end_index} onto cropped image.")
        _font_size = font_size or self.default_font_size
        try:
            # Ensure image is suitable for drawing
            if image_object.mode not in ('RGB', 'RGBA'): image_object = image_object.convert("RGBA")
            elif image_object.mode == 'RGB' and (any(c[3] < 255 for c in [line_color, text_color]) or (text_bg_color and text_bg_color[3] < 255)):
                image_object = image_object.convert("RGBA")

            draw = ImageDraw.Draw(image_object, image_object.mode)
            font = self._load_font(font_path, fallback_font_path, _font_size)
            img_width, img_height = image_object.size
            min_x_crop, min_y_crop = crop_box[0], crop_box[1]

            for i in range(start_index, end_index):
                line = lines_data_full_page[i]
                coords = line.get('baseline_coords')
                original_line_number = i + 1
                line_number_str = str(original_line_number)

                if coords and len(coords) >= 2:
                    translated_coords = []
                    for p in coords:
                        try:
                            tx = int(round(float(p[0]) - min_x_crop))
                            ty = int(round(float(p[1]) - min_y_crop))
                            translated_coords.append((tx, ty))
                        except (TypeError, ValueError, IndexError):
                            logger.warning(f"Skipping point {p} in line {original_line_number} during chunk drawing translation.")
                            translated_coords = []; break # Invalidate line

                    if len(translated_coords) >= 2:
                        draw.line(translated_coords, fill=line_color, width=line_width)
                        start_point_translated = translated_coords[0]
                        self._draw_single_number(draw, line_number_str, start_point_translated, font, _font_size, text_color, text_bg_color, text_offset_x, text_offset_y, img_width, img_height, 'left')
                        end_point_translated = translated_coords[-1]
                        self._draw_single_number(draw, line_number_str, end_point_translated, font, _font_size, text_color, text_bg_color, text_offset_x, text_offset_y, img_width, img_height, 'right')
                    else: logger.debug(f"Skipping drawing line {original_line_number} in chunk: Not enough valid translated points.")
                else: logger.debug(f"Skipping line {original_line_number} in chunk drawing: Invalid/missing baseline_coords.")

            logger.debug(f"Finished drawing lines/numbers for chunk {start_index+1}-{end_index}.")
            return image_object

        except (FileNotFoundError, IOError, ValueError) as e:
            logger.error(f"Error during chunk baseline/number drawing: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in draw_chunk_baselines_numbers: {e}", exc_info=True)
            return None

    def calculate_average_baseline_angle(self, lines_data: List[Dict[str, Any]]) -> float:
        """Calculates the average angle (radians) of baselines using vector averaging."""
        # (Logic copied from original script, unchanged)
        sum_vector_x = 0.0; sum_vector_y = 0.0; valid_baseline_count = 0
        for line in lines_data:
            coords = line.get('baseline_coords')
            if coords and len(coords) >= 2:
                start_point = coords[0]; end_point = coords[-1]
                try: x1, y1 = float(start_point[0]), float(start_point[1]); x2, y2 = float(end_point[0]), float(end_point[1])
                except (TypeError, IndexError, ValueError): logger.warning(f"Skipping angle calc for line {line.get('id', 'N/A')}: invalid coords {coords}"); continue
                dx = x2 - x1; dy = y2 - y1
                length = math.hypot(dx, dy) # math.hypot is equivalent to sqrt(dx*dx + dy*dy)
                if length > 1e-6: sum_vector_x += dx / length; sum_vector_y += dy / length; valid_baseline_count += 1
                else: logger.debug(f"Skipping angle calc for line {line.get('id', 'N/A')}: zero length.")
        if valid_baseline_count == 0: logger.warning("No valid baselines found to calculate average angle. Returning 0.0."); return 0.0
        average_angle_rad = math.atan2(sum_vector_y, sum_vector_x)
        logger.info(f"Calculated average baseline angle: {math.degrees(average_angle_rad):.2f} deg ({average_angle_rad:.4f} rad) from {valid_baseline_count} baselines.")
        return average_angle_rad

    def _rotate_point(self, point: Tuple[float, float], center: Tuple[float, float], angle_rad: float) -> Tuple[float, float]:
        """Internal helper to rotate a single point."""
        x, y = point; cx, cy = center; cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        translated_x = x - cx; translated_y = y - cy
        rotated_x = translated_x * cos_a - translated_y * sin_a; rotated_y = translated_x * sin_a + translated_y * cos_a
        final_x = rotated_x + cx; final_y = rotated_y + cy
        return final_x

    def rotate_image_and_coords(
        self, image_bytes: bytes, lines_data: List[Dict[str, Any]],
        image_width: int, image_height: int
    ) -> Tuple[Optional[bytes], Optional[List[Dict[str, Any]]], Optional[Tuple[int, int]]]:
        """
        Rotates image content and transforms baseline coordinates.

        Args:
            image_bytes: Original image bytes.
            lines_data: List of line data dictionaries with 'baseline_coords'.
            image_width: Original image width.
            image_height: Original image height.

        Returns:
            Tuple: (rotated_image_bytes, transformed_lines_data, new_dimensions) or (None, None, None) on error.
                   new_dimensions is (new_width, new_height).
        """
        logger.info("Rotating image and transforming coordinates...")
        avg_angle_rad = self.calculate_average_baseline_angle(lines_data)
        negligible_angle_degrees = 0.2
        if abs(math.degrees(avg_angle_rad)) < negligible_angle_degrees:
            logger.info("Rotation angle negligible. Returning original data.")
            return image_bytes, lines_data, (image_width, image_height)

        # Rotate image BY the average angle to make baselines horizontal
        rotation_angle_rad = avg_angle_rad
        rotation_angle_deg = math.degrees(rotation_angle_rad)
        logger.info(f"Rotating image by {rotation_angle_deg:.2f} degrees.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            original_center = (image_width / 2.0, image_height / 2.0)
            if image.mode != 'RGBA': image = image.convert('RGBA')

            rotated_image = image.rotate(
                rotation_angle_deg, resample=Image.Resampling.BICUBIC, expand=True,
                center=None, fillcolor=(255, 255, 255, 255) # White background
            )
            new_width, new_height = rotated_image.size
            new_center = (new_width / 2.0, new_height / 2.0)
            logger.info(f"Rotated image dimensions: {new_width}x{new_height}")

            # Transform coordinates
            transformed_lines = []
            cos_a = math.cos(rotation_angle_rad); sin_a = math.sin(rotation_angle_rad)
            for line in lines_data:
                new_line = line.copy()
                original_coords = line.get('baseline_coords')
                if original_coords:
                    transformed_coords = []
                    for p_orig in original_coords:
                        try:
                            p_orig_float = (float(p_orig[0]), float(p_orig[1]))
                            p_rel_x = p_orig_float[0] - original_center[0]
                            p_rel_y = p_orig_float[1] - original_center[1]
                            p_rel_rot_x = p_rel_x * cos_a - p_rel_y * sin_a
                            p_rel_rot_y = p_rel_x * sin_a + p_rel_y * cos_a
                            p_new_x = p_rel_rot_x + new_center[0]
                            p_new_y = p_rel_rot_y + new_center[1]
                            transformed_coords.append((p_new_x, p_new_y))
                        except (TypeError, IndexError, ValueError) as p_err:
                            logger.warning(f"Skipping point {p_orig} in line {line.get('id', 'N/A')} during rotation: {p_err}.")
                            transformed_coords = None; break # Invalidate line if any point fails
                    new_line['baseline_coords'] = transformed_coords
                transformed_lines.append(new_line)

            # Prepare output bytes (JPEG recommended for LLMs)
            output_buffer = io.BytesIO()
            image_to_save = rotated_image
            if image_to_save.mode == 'RGBA':
                bg = Image.new("RGB", image_to_save.size, (255, 255, 255))
                try: bg.paste(image_to_save, mask=image_to_save.split()[3])
                except IndexError: bg.paste(image_to_save)
                image_to_save = bg
            elif image_to_save.mode != 'RGB': image_to_save = image_to_save.convert('RGB')
            image_to_save.save(output_buffer, format="JPEG", quality=95)
            rotated_image_bytes = output_buffer.getvalue()

            logger.info("Image rotation and coordinate transformation successful.")
            return rotated_image_bytes, transformed_lines, (new_width, new_height)

        except Exception as e:
            logger.error(f"Error during image rotation/coordinate transformation: {e}", exc_info=True)
            return None, None, None

    def segment_image(self, image_path: str, output_dir: str, base_filename: str) -> Optional[List[str]]:
        """Segments an image into four fixed, overlapping segments and saves them."""
        logger.info(f"Segmenting image '{image_path}' into directory '{output_dir}'")
        try:
            original_image = Image.open(image_path)
        except FileNotFoundError: raise ImageProcessingError(f"Image file not found: {image_path}")
        except Exception as e: raise ImageProcessingError(f"Error opening image {image_path}: {e}") from e

        width, height = original_image.size
        segment_width = 1704; segment_height = 674 # Fixed dimensions from template
        if width != segment_width: logger.warning(f"Input width {width} != expected {segment_width}.")
        if height < (1527 + segment_height): logger.warning(f"Input height {height} too small for final segment.")

        segment_coordinates = [
            (0, 0, segment_width, segment_height), (0, 509, segment_width, 509 + segment_height),
            (0, 1018, segment_width, 1018 + segment_height), (0, 1527, segment_width, 1527 + segment_height)
        ]

        try: os.makedirs(output_dir, exist_ok=True)
        except OSError as e: raise ImageProcessingError(f"Failed to create output directory '{output_dir}': {e}") from e

        saved_segment_paths = []
        for i, coords in enumerate(segment_coordinates):
            left, top, right, bottom = coords
            actual_right = min(right, width); actual_bottom = min(bottom, height)
            if left >= actual_right or top >= actual_bottom: logger.warning(f"Skipping segment {i+1}: invalid dimensions."); continue

            try:
                segment = original_image.crop((left, top, actual_right, actual_bottom))
                output_filename = os.path.join(output_dir, f"{base_filename}_segment_{i+1}.png") # Save as PNG
                segment.save(output_filename, "PNG")
                logger.info(f"Segment {i+1} saved to: {output_filename}")
                saved_segment_paths.append(output_filename)
            except Exception as e: logger.error(f"Failed to crop/save segment {i+1}: {e}")

        if len(saved_segment_paths) == 4: return saved_segment_paths
        else:
            logger.error("Segmentation failed: Not all segments were saved.")
            # Clean up partial segments
            for path in saved_segment_paths:
                try: os.remove(path)
                except OSError: pass
            return None

    def image_to_base64(self, image_object: Image.Image, format: str = "JPEG") -> str:
        """Converts a PIL Image object to a base64 encoded string."""
        buffered = io.BytesIO()
        save_format = format.upper()
        image_to_save = image_object # Start with the input object

        try:
            if save_format == "JPEG":
                if image_object.mode == 'RGBA':
                    logger.debug("Converting RGBA to RGB for JPEG base64 encoding.")
                    bg = Image.new("RGB", image_object.size, (255, 255, 255))
                    try: bg.paste(image_object, mask=image_object.split()[3])
                    except IndexError: bg.paste(image_object)
                    image_to_save = bg
                elif image_object.mode != 'RGB': image_to_save = image_object.convert('RGB')
                image_to_save.save(buffered, format="JPEG", quality=95)
            elif save_format == "PNG":
                # PNG supports RGBA, save directly
                image_object.save(buffered, format="PNG")
            else:
                # Attempt other formats, fallback to PNG
                try: image_object.save(buffered, format=save_format)
                except OSError: logger.warning(f"Format {save_format} failed, saving as PNG."); image_object.save(buffered, format="PNG")

            img_byte = buffered.getvalue()
            img_base64 = base64.b64encode(img_byte).decode('utf-8')
            return img_base64
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert image to base64 ({save_format}): {e}") from e

    def calculate_bounding_box(self, lines_data: List[Dict[str, Any]], start_index: int, end_index: int, padding: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculates the bounding box for a range of lines with padding."""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        found_coords = False
        start_index = max(0, start_index)
        end_index = min(len(lines_data), end_index)
        if start_index >= end_index: return None

        for i in range(start_index, end_index):
            coords = lines_data[i].get('baseline_coords')
            if coords:
                for p in coords:
                    try: x, y = float(p[0]), float(p[1]); min_x=min(min_x,x); min_y=min(min_y,y); max_x=max(max_x,x); max_y=max(max_y,y); found_coords=True
                    except (TypeError, IndexError, ValueError): logger.warning(f"Invalid coord {p} in line {i}. Skipping."); continue
        if not found_coords: return None

        pad_min_x = max(0, int(min_x - padding)); pad_min_y = max(0, int(min_y - padding * 2))
        pad_max_x = int(max_x + padding); pad_max_y = int(max_y + padding * 2)
        if pad_max_x <= pad_min_x: pad_max_x = pad_min_x + 1
        if pad_max_y <= pad_min_y: pad_max_y = pad_min_y + 1
        return (pad_min_x, pad_min_y, pad_max_x, pad_max_y)