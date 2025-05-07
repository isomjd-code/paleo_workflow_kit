# paleo_workflow_kit/page_xml_handler.py

import xml.etree.ElementTree as ET
import io
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union

# Get logger for this module
logger = logging.getLogger(__name__)

# Define PAGE XML Namespace constant
PAGE_XML_NAMESPACE = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
# Register globally for cleaner output and finding elements without explicit nsmap
ET.register_namespace('', PAGE_XML_NAMESPACE['page'])

class PageXMLParseError(Exception):
    """Custom exception for errors during PAGE XML parsing."""
    pass

class LineNotFoundError(Exception):
    """Custom exception when a specific TextLine cannot be found."""
    pass

class PageXMLHandler:
    """
    Handles parsing, analysis, and modification of PAGE XML content.

    Operates on an internal ElementTree representation of the XML.
    Modifications are performed in place on the internal tree.
    Use get_xml_bytes() to retrieve the modified XML content.
    """

    def __init__(self, xml_content_bytes: bytes):
        """
        Initializes the handler by parsing the provided XML bytes.

        Args:
            xml_content_bytes: The PAGE XML content as bytes.

        Raises:
            PageXMLParseError: If the XML cannot be parsed.
            ValueError: If xml_content_bytes is empty or not bytes.
        """
        if not isinstance(xml_content_bytes, bytes) or not xml_content_bytes:
            raise ValueError("xml_content_bytes must be a non-empty bytes object.")

        self.original_bytes = xml_content_bytes
        self.namespace = PAGE_XML_NAMESPACE # Store for potential use
        self._lines_data_cache: Optional[List[Dict[str, Any]]] = None # Cache for parsed line data

        try:
            # Use io.BytesIO to parse directly from bytes
            xml_file_like = io.BytesIO(self.original_bytes)
            self.tree: ET.ElementTree = ET.parse(xml_file_like)
            self.root: ET.Element = self.tree.getroot()
            logger.debug("Successfully parsed PAGE XML content.")
        except ET.ParseError as e:
            logger.error(f"Failed to parse PAGE XML: {e}")
            try:
                # Log the beginning of the problematic XML for debugging
                logger.error(f"Problematic XML content (first 500 bytes):\n{xml_content_bytes[:500].decode('utf-8', errors='replace')}")
            except Exception:
                 logger.error("Could not decode problematic XML content for logging.")
            raise PageXMLParseError(f"Failed to parse PAGE XML: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during XML initialization: {e}", exc_info=True)
            raise PageXMLParseError(f"Unexpected error during XML initialization: {e}") from e

    def _parse_lines_data(self) -> List[Dict[str, Any]]:
        """
        Internal helper to parse line data (ID, text, coords, index).
        Caches the result.
        """
        if self._lines_data_cache is not None:
            return self._lines_data_cache

        logger.debug("Parsing detailed line data from XML tree...")
        lines_data = []
        line_counter = 0
        # Find all TextLine elements regardless of parent first to ensure correct indexing
        # This is crucial for matching external lists like uncertainty_flags
        all_text_lines = self.root.findall('.//page:TextLine', self.namespace)
        logger.debug(f"Found {len(all_text_lines)} TextLine elements in total.")

        for idx, line in enumerate(all_text_lines):
            line_id = line.get('id')
            baseline_elem = line.find('page:Baseline', self.namespace)
            textequiv_elem = line.find('page:TextEquiv', self.namespace)

            baseline_points_str = baseline_elem.get('points') if baseline_elem is not None else None
            htr_text = ""
            unicode_elem = textequiv_elem.find('page:Unicode', self.namespace) if textequiv_elem else None
            if unicode_elem is not None and unicode_elem.text:
                htr_text = unicode_elem.text.strip()

            baseline_coords = None
            if baseline_points_str:
                try:
                    baseline_coords = [tuple(map(int, p.split(','))) for p in baseline_points_str.split()]
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse baseline points for line ID '{line_id}' (Index {idx}): '{baseline_points_str}'")

            # Ensure every line gets a unique ID for reliable mapping later
            # If an ID exists, use it. Otherwise, generate one based on sequence.
            # This generated ID is primarily for internal mapping if the original XML lacks IDs.
            temp_id = line_id if line_id else f"seqid_{idx}"
            if not line_id:
                logger.debug(f"Line at index {idx} has no ID. Assigning temporary internal ID '{temp_id}'.")
                # Note: We don't modify the XML tree here, just use temp_id for the data structure

            lines_data.append({
                'id': temp_id, # Use original ID if present, otherwise generated seq ID
                'original_id': line_id, # Store the original ID (or None) separately
                'baseline_coords': baseline_coords,
                'htr_text': htr_text,
                'seq_index': idx # Store 0-based sequence index
            })
            line_counter += 1 # Keep track just for logging

        logger.info(f"Successfully parsed detailed data for {line_counter} lines.")
        self._lines_data_cache = lines_data
        return self._lines_data_cache

    def _find_line_element_by_index(self, target_index: int) -> Optional[ET.Element]:
        """Finds a TextLine element based on its sequential index in the document."""
        all_text_lines = self.root.findall('.//page:TextLine', self.namespace)
        if 0 <= target_index < len(all_text_lines):
            return all_text_lines[target_index]
        else:
            logger.warning(f"Could not find TextLine element at sequential index {target_index}.")
            return None

    def _find_line_element(self, line_identifier: Union[str, int]) -> Optional[ET.Element]:
        """
        Finds a TextLine element by its ID attribute or sequential index.
        Prefers ID match if identifier is string, uses index if integer.
        """
        if isinstance(line_identifier, str):
            # Try finding by ID first
            xpath_query = f".//page:TextLine[@id='{line_identifier}']"
            element = self.root.find(xpath_query, self.namespace)
            if element is not None:
                return element
            else:
                # Fallback: Check if the string looks like our generated seqid_
                if line_identifier.startswith("seqid_"):
                    try:
                        index = int(line_identifier.split('_')[1])
                        return self._find_line_element_by_index(index)
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse index from generated ID '{line_identifier}'.")
                        return None
                else:
                    logger.warning(f"Could not find TextLine element with ID '{line_identifier}'.")
                    return None
        elif isinstance(line_identifier, int):
            # Find by sequential index
            return self._find_line_element_by_index(line_identifier)
        else:
            logger.error(f"Invalid line_identifier type: {type(line_identifier)}. Must be str (ID) or int (index).")
            return None

    def get_lines_data(self) -> List[Dict[str, Any]]:
        """
        Returns detailed data for each TextLine.
        Result is cached after first call.

        Returns:
            List of dictionaries, each containing:
            'id': The original ID attribute or a generated sequential ID ('seqid_N').
            'original_id': The original ID attribute from the XML (can be None).
            'baseline_coords': List of (x, y) tuples or None.
            'htr_text': Stripped text content from Unicode element or "".
            'seq_index': 0-based sequential index of the line in the document.
        """
        return self._parse_lines_data()

    def get_text_lines(self) -> List[str]:
        """Returns a list of HTR text strings in document order."""
        lines_data = self.get_lines_data()
        return [line.get('htr_text', '') for line in lines_data]

    def get_page_dimensions(self) -> Optional[Tuple[int, int]]:
        """Returns the page width and height from the Page element."""
        page_elem = self.root.find('.//page:Page', self.namespace)
        if page_elem is not None:
            width_str = page_elem.get('imageWidth')
            height_str = page_elem.get('imageHeight')
            if width_str and height_str:
                try:
                    return int(width_str), int(height_str)
                except (ValueError, TypeError):
                    logger.error(f"Could not convert page dimensions to integers: W='{width_str}', H='{height_str}'")
                    return None
        logger.error("Could not find Page element or its dimensions.")
        return None

    def update_text_lines(self, line_updates: Dict[Union[str, int], str]) -> Tuple[int, int]:
        """
        Updates the Unicode text content of specified TextLine elements.

        Args:
            line_updates: A dictionary where keys are line IDs (str) or
                          sequential indices (int) and values are the new text strings.

        Returns:
            A tuple: (number_of_lines_updated, number_of_lines_failed).
        """
        updated_count = 0
        failed_count = 0
        logger.info(f"Attempting to update text for {len(line_updates)} lines...")

        for identifier, new_text in line_updates.items():
            text_line_element = self._find_line_element(identifier)
            if text_line_element is None:
                logger.warning(f"Could not find line with identifier '{identifier}' to update text.")
                failed_count += 1
                continue

            text_equiv = text_line_element.find('page:TextEquiv', self.namespace)
            if text_equiv is None:
                # Create TextEquiv if it doesn't exist
                text_equiv = ET.SubElement(text_line_element, '{%s}TextEquiv' % self.namespace['page'])
                logger.debug(f"Created missing TextEquiv for line '{identifier}'.")

            unicode_elem = text_equiv.find('page:Unicode', self.namespace)
            if unicode_elem is None:
                # Create Unicode element if it doesn't exist
                unicode_elem = ET.SubElement(text_equiv, '{%s}Unicode' % self.namespace['page'])
                logger.debug(f"Created missing Unicode element for line '{identifier}'.")

            # Update text content
            if unicode_elem.text != new_text:
                unicode_elem.text = new_text
                updated_count += 1
                logger.debug(f"Updated text for line '{identifier}'.")
            # else: logger.debug(f"Text for line '{identifier}' already matches. No update needed.")

        logger.info(f"Text update complete. Updated: {updated_count}, Failed/Not Found: {failed_count}.")
        return updated_count, failed_count

    def apply_unclear_tags(self, uncertainty_flags: List[bool]) -> Tuple[int, int]:
        """
        Adds or removes 'unclear {offset...}' tag in the 'custom' attribute
        of TextLine elements based on a boolean list matching line sequence.

        Args:
            uncertainty_flags: A list of booleans. True means the line at that
                               index should be marked unclear, False means any
                               existing unclear tag should be removed. Length
                               must match the number of lines in the document.

        Returns:
            A tuple: (number_of_tags_added, number_of_tags_removed).
        """
        lines_data = self.get_lines_data() # Get ordered line data
        num_lines_xml = len(lines_data)
        num_flags = len(uncertainty_flags)

        if num_lines_xml != num_flags:
            logger.error(f"Cannot apply unclear tags: Mismatch between XML lines ({num_lines_xml}) and uncertainty flags ({num_flags}).")
            # Consider raising an error or returning specific failure codes
            return 0, 0

        tags_added = 0
        tags_removed = 0
        logger.info(f"Applying unclear tags based on {num_flags} flags...")

        # Regex to find/remove the specific unclear tag format robustly
        unclear_pattern = re.compile(r"unclear\s*\{offset:\d+;\s*length:\d+;\}\s*", re.IGNORECASE)

        for idx, line_info in enumerate(lines_data):
            is_uncertain = uncertainty_flags[idx]
            # Use sequential index to find the element reliably, as IDs might be missing/inconsistent
            text_line_element = self._find_line_element_by_index(idx)

            if text_line_element is None:
                logger.warning(f"Could not find TextLine element at index {idx} to apply/remove unclear tag.")
                continue

            custom_attr = text_line_element.get('custom', '')
            line_text = line_info.get('htr_text', '') # Get text from parsed data
            current_length = len(line_text) if line_text else 0
            tag_content_to_match_or_add = f"unclear {{offset:0; length:{current_length};}}"

            tag_match = unclear_pattern.search(custom_attr)
            tag_exists = tag_match is not None

            if is_uncertain:
                # Add tag if it doesn't exist
                if not tag_exists:
                    new_custom_attr = (custom_attr.strip() + " " + tag_content_to_match_or_add).strip()
                    text_line_element.set('custom', new_custom_attr)
                    tags_added += 1
                    logger.debug(f"Line {idx+1} (ID: {line_info['id']}): Added 'unclear' tag.")
                # else: logger.debug(f"Line {idx+1}: Is uncertain, tag already exists.")
            else:
                # Remove tag if it exists
                if tag_exists:
                    new_custom_attr = unclear_pattern.sub('', custom_attr).strip()
                    if new_custom_attr:
                        text_line_element.set('custom', new_custom_attr)
                    else: # Remove attribute if it becomes empty
                        if 'custom' in text_line_element.attrib:
                            del text_line_element.attrib['custom']
                    tags_removed += 1
                    logger.debug(f"Line {idx+1} (ID: {line_info['id']}): Removed 'unclear' tag.")
                # else: logger.debug(f"Line {idx+1}: Is certain, tag not present.")

        logger.info(f"Unclear tag processing complete. Tags Added: {tags_added}, Tags Removed: {tags_removed}.")
        return tags_added, tags_removed

    def remove_empty_regions(self) -> int:
        """Finds and removes TextRegions containing no lines with text content."""
        logger.info("Attempting to remove empty TextRegions...")
        regions_deleted_count = 0
        page = self.root.find('.//page:Page', self.namespace)
        if page is None:
            logger.warning("No Page element found. Cannot remove empty regions.")
            return 0

        regions_to_remove = []
        for region in page.findall('./page:TextRegion', self.namespace):
            region_id = region.get('id', 'N/A')
            has_text_content = False
            # Check descendant TextLines for any text
            for line in region.findall('.//page:TextLine', self.namespace):
                unicode_elem = line.find('.//page:Unicode', self.namespace)
                if unicode_elem is not None and unicode_elem.text and unicode_elem.text.strip():
                    has_text_content = True
                    break
            # Also check direct TextEquiv on region itself
            if not has_text_content:
                 region_unicode = region.find('./page:TextEquiv/page:Unicode', self.namespace)
                 if region_unicode is not None and region_unicode.text and region_unicode.text.strip():
                      has_text_content = True

            if not has_text_content:
                regions_to_remove.append(region)
                logger.debug(f"Marking empty region ID '{region_id}' for deletion.")

        for region in regions_to_remove:
            try:
                page.remove(region)
                regions_deleted_count += 1
            except ValueError:
                logger.warning(f"Could not remove region ID '{region.get('id', 'N/A')}' - already removed?")

        logger.info(f"Empty region removal complete. Deleted {regions_deleted_count} regions.")
        return regions_deleted_count

    def remove_short_lines(self, min_length: int) -> int:
        """Finds and removes TextLines with text shorter than min_length or empty."""
        logger.info(f"Attempting to remove lines shorter than {min_length} characters (or empty)...")
        lines_deleted_count = 0
        # Find potential parent elements (Page or TextRegion)
        parents = self.root.findall('.//page:TextRegion', self.namespace)
        page_element = self.root.find('.//page:Page', self.namespace)
        if not parents and page_element is not None:
            parents = [page_element]
        elif not parents:
             logger.warning("No Page or TextRegion elements found. Cannot remove short lines.")
             return 0

        for parent in parents:
            lines_to_remove = []
            for line in parent.findall('./page:TextLine', self.namespace):
                unicode_elem = line.find('.//page:Unicode', self.namespace)
                line_text = unicode_elem.text.strip() if unicode_elem is not None and unicode_elem.text else ""
                if len(line_text) < min_length:
                    lines_to_remove.append(line)
                    logger.debug(f"Marking line ID '{line.get('id', 'N/A')}' for deletion (length {len(line_text)} < {min_length}).")

            for line in lines_to_remove:
                try:
                    parent.remove(line)
                    lines_deleted_count += 1
                except ValueError:
                    logger.warning(f"Could not remove short line ID '{line.get('id', 'N/A')}' - already removed?")

        logger.info(f"Short line removal complete. Deleted {lines_deleted_count} lines.")
        return lines_deleted_count

    def remove_unclear_tags_from_custom(self) -> int:
        """Removes 'unclear {offset...}' tags from all TextLine custom attributes."""
        logger.info("Attempting to remove all 'unclear {offset...}' tags...")
        tags_removed_count = 0
        unclear_pattern = re.compile(r"unclear\s*\{offset:\d+;\s*length:\d+;\}\s*", re.IGNORECASE)

        for text_line in self.root.findall('.//page:TextLine', self.namespace):
            custom_attr = text_line.get('custom')
            if custom_attr and 'unclear {offset:' in custom_attr:
                original_attr = custom_attr
                modified_attr = unclear_pattern.sub(' ', custom_attr).strip() # Replace with space, then strip
                if modified_attr != original_attr:
                    if modified_attr:
                        text_line.set('custom', modified_attr)
                    else:
                        if 'custom' in text_line.attrib:
                            del text_line.attrib['custom']
                    tags_removed_count += 1
                    logger.debug(f"Removed unclear tag from line ID '{text_line.get('id', 'N/A')}'.")

        logger.info(f"Unclear tag removal complete. Removed tags from {tags_removed_count} lines.")
        return tags_removed_count

    def find_and_replace(self, find_string: str, replace_string: str, case_sensitive: bool = True) -> int:
        """Performs find and replace on Unicode text of all TextLines."""
        if not find_string:
            logger.warning("Find string is empty. Skipping this replacement rule.")
            return 0

        logger.info(f"Applying rule: Find='{find_string}', Replace='{replace_string}', CaseSensitive={case_sensitive}")
        replacements_made_count = 0

        for line in self.root.findall('.//page:TextLine', self.namespace):
            unicode_element = line.find('./page:TextEquiv/page:Unicode', self.namespace)
            if unicode_element is not None and unicode_element.text:
                original_text = unicode_element.text
                modified_text = None
                replacements_in_line = 0

                if case_sensitive:
                    if find_string in original_text:
                        replacements_in_line = original_text.count(find_string)
                        modified_text = original_text.replace(find_string, replace_string)
                else:
                    # Simple case-insensitive replace using regex
                    try:
                        # Count occurrences first (case-insensitive)
                        replacements_in_line = len(re.findall(re.escape(find_string), original_text, re.IGNORECASE))
                        if replacements_in_line > 0:
                             modified_text = re.sub(re.escape(find_string), replace_string, original_text, flags=re.IGNORECASE)
                    except re.error as re_err:
                         logger.error(f"Regex error for find='{find_string}': {re_err}. Skipping rule for this line.")
                         continue


                if modified_text is not None and modified_text != original_text:
                    unicode_element.text = modified_text
                    replacements_made_count += replacements_in_line
                    # logger.debug(f"Replaced in Line ID '{line.get('id', 'N/A')}': '{original_text[:50]}...' -> '{modified_text[:50]}...'")

        logger.info(f"Rule ('{find_string}' -> '{replace_string}') made {replacements_made_count} replacements.")
        return replacements_made_count

    def add_metadata_item(self, type_val: str, name_val: str, value_val: str) -> bool:
        """Adds a MetadataItem to the Page/Metadata section."""
        logger.info(f"Adding metadata: Type='{type_val}', Name='{name_val}', Value='{value_val}'")
        metadata_xpath = './/page:Metadata'
        metadata = self.root.find(metadata_xpath, self.namespace)

        if metadata is None:
            page_elem = self.root.find('.//page:Page', self.namespace)
            if page_elem is None:
                logger.error("Cannot add metadata: No <Page> element found.")
                return False
            logger.info("No <Metadata> element found. Creating one under <Page>.")
            metadata = ET.SubElement(page_elem, '{%s}Metadata' % self.namespace['page'])

        # Add the new item
        try:
            ET.SubElement(metadata, '{%s}MetadataItem' % self.namespace['page'], attrib={
                'type': type_val,
                'name': name_val,
                'value': value_val
            })
            logger.debug("Successfully added MetadataItem.")
            return True
        except Exception as e:
            logger.error(f"Failed to add MetadataItem: {e}")
            return False

    def get_xml_bytes(self) -> bytes:
        """Serializes the current internal XML tree back to bytes."""
        logger.debug("Serializing internal XML tree to bytes...")
        try:
            # Apply indentation for readability if Python version supports it
            if hasattr(ET, 'indent'):
                ET.indent(self.tree, space="  ", level=0) # Use 2 spaces for indent

            output_buffer = io.BytesIO()
            self.tree.write(output_buffer, encoding='utf-8', xml_declaration=True)
            modified_xml_bytes = output_buffer.getvalue()
            logger.debug(f"XML serialization successful ({len(modified_xml_bytes)} bytes).")
            return modified_xml_bytes
        except Exception as e:
            logger.error(f"Failed to serialize XML tree to bytes: {e}", exc_info=True)
            # Maybe return original bytes as fallback? Or raise error?
            # Returning original might hide modification errors. Raising is safer.
            raise IOError("Failed to serialize modified XML tree.") from e

    @classmethod
    def from_file(cls, xml_filepath: str) -> 'PageXMLHandler':
        """Class method to initialize the handler from an XML file path."""
        logger.info(f"Initializing PageXMLHandler from file: {xml_filepath}")
        try:
            with open(xml_filepath, 'rb') as f:
                xml_bytes = f.read()
            return cls(xml_bytes)
        except FileNotFoundError:
            logger.error(f"XML file not found: {xml_filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading XML file {xml_filepath}: {e}")
            raise