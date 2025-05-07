# paleo_workflow_kit/transkribus_api.py

import requests
import xml.etree.ElementTree as ET
import os
import logging
import time
import json
import io
from typing import List, Dict, Any, Optional, Tuple, Union

# Get logger for this module
logger = logging.getLogger(__name__)

# Define PAGE XML Namespace constant (can be imported if defined elsewhere)
PAGE_XML_NAMESPACE = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

class TranskribusAPI:
    """
    A client class for interacting with the Transkribus REST API.

    Handles common operations like listing documents/pages, downloading/uploading
    XML, triggering HTR, and deleting resources. Requires an authenticated
    requests.Session object for initialization.
    """

    def __init__(self, session: requests.Session, base_url: str, request_delay: float = 0.5):
        """
        Initializes the TranskribusAPI client.

        Args:
            session: An authenticated requests.Session object (already logged in).
            base_url: The base URL for the Transkribus REST API
                      (e.g., "https://transkribus.eu/TrpServer/rest").
            request_delay: Default delay in seconds between API calls that modify data
                           or could be rate-limited.
        """
        if not isinstance(session, requests.Session):
            raise TypeError("session must be an authenticated requests.Session object")
        self.session = session
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.request_delay = request_delay
        logger.info(f"TranskribusAPI client initialized with base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Internal helper to make requests with error handling and delay."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        verb = method.upper()
        # Add delay for modifying requests or potentially sensitive lists
        is_modifying = verb in ["POST", "PUT", "DELETE"]
        is_list = "list" in endpoint.lower() or "pages" in endpoint.lower()

        if is_modifying or is_list:
            logger.debug(f"Applying request delay: {self.request_delay}s")
            time.sleep(self.request_delay)

        response = None
        try:
            logger.debug(f"Making {verb} request to: {url} with params: {kwargs.get('params')}")
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            error_text = http_err.response.text[:500] if http_err.response is not None else "N/A"
            logger.error(f"HTTP error {status_code} for {verb} {url}: {http_err}")
            logger.error(f"--> Response Body Snippet: {error_text}")
            # Specific handling for common errors
            if status_code == 401: logger.error("--> Authentication failed (401 Unauthorized). Check credentials/session.")
            elif status_code == 403: logger.error("--> Permission denied (403 Forbidden). Check user permissions for the resource.")
            elif status_code == 404: logger.error("--> Resource not found (404). Check IDs (collection, document, page).")
            elif status_code == 500: logger.error("--> Internal Server Error (500). Check request payload/parameters or Transkribus status.")
            return None # Indicate failure
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred for {verb} {url}")
            return None
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Network/Request error for {verb} {url}: {req_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during request to {verb} {url}: {e}", exc_info=True)
            return None

    def list_documents(self, coll_id: int) -> List[Dict[str, Any]]:
        """Retrieves a list of all documents (docId, title) in a collection."""
        logger.info(f"Retrieving document list for Collection ID: {coll_id}")
        all_documents = []
        offset = 0
        limit = 100
        total_docs = -1

        while True:
            endpoint = f"collections/{coll_id}/list"
            params = {'start': offset, 'n': limit}
            response = self._make_request("GET", endpoint, params=params, timeout=60)

            if response is None:
                logger.warning("Failed to fetch document list chunk. Returning potentially incomplete list.")
                break

            try:
                data = response.json()
                docs_in_page = []
                current_page_count = 0
                is_direct_list_response = False

                # Robust parsing (copied from original script)
                if isinstance(data, list): docs_in_page = data; current_page_count = len(docs_in_page); is_direct_list_response = True
                elif isinstance(data, dict):
                    if 'pageList' in data and isinstance(data.get('pageList'), dict) and isinstance(data['pageList'].get('docs'), list):
                        docs_in_page = data['pageList']['docs']; current_page_count = len(docs_in_page)
                        if isinstance(data.get('md'), dict): total_docs = int(data['md'].get('nrOfHits', -1))
                    elif 'documents' in data and isinstance(data.get('documents'), list):
                        docs_in_page = data['documents']; current_page_count = len(docs_in_page)
                        if 'total' in data: total_docs = int(data.get('total', -1))
                    elif 'docs' in data and isinstance(data.get('docs'), list): # Another structure seen
                        docs_in_page = data['docs']; current_page_count = len(docs_in_page)
                        if 'nrOfHits' in data: total_docs = int(data.get('nrOfHits', -1))
                    else: logger.warning(f"Unexpected JSON dict structure in doc list: {list(data.keys())}"); break
                else: logger.error(f"Unexpected response type from doc list: {type(data)}"); break

                if not docs_in_page and offset == 0: logger.warning(f"No documents found in collection {coll_id}."); break

                for doc in docs_in_page:
                    if isinstance(doc, dict) and 'docId' in doc and 'title' in doc:
                        try: all_documents.append({'docId': int(doc['docId']), 'title': doc['title'], 'uploadTimestamp': doc.get('uploadTimestamp')})
                        except (ValueError, TypeError): logger.warning(f"Skipping doc due to invalid docId format: {doc}")
                    else: logger.warning(f"Skipping doc entry with missing fields: {doc}")

                logger.debug(f"Fetched {len(docs_in_page)} docs (Total: {len(all_documents)}).")

                if is_direct_list_response and offset == 0: break
                if current_page_count < limit: break
                if total_docs != -1 and len(all_documents) >= total_docs: break
                if current_page_count == 0 and offset > 0: break

                offset += limit

            except json.JSONDecodeError as e:
                logger.error(f"Could not decode JSON response for doc list: {e}")
                logger.error(f"Response text: {response.text[:500]}")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing doc list chunk: {e}", exc_info=True)
                break

        logger.info(f"Finished fetching documents for Coll {coll_id}. Total retrieved: {len(all_documents)}")
        return all_documents

    def list_pages(self, coll_id: int, doc_id: int) -> List[Dict[str, Any]]:
        """Retrieves details (pageNr, pageId, imgFileName) for ALL pages in a document."""
        logger.debug(f"Retrieving all page details for Doc ID: {doc_id}")
        all_pages = []
        offset = 0
        limit = 100

        while True:
            endpoint = f"collections/{coll_id}/{doc_id}/pages"
            params = {'start': offset, 'n': limit}
            response = self._make_request("GET", endpoint, params=params, timeout=60)

            if response is None:
                logger.warning(f"Failed to fetch page list chunk for Doc {doc_id}. Returning potentially incomplete list.")
                break

            try:
                data = response.json()
                pages_in_batch = []
                # Handle potential response structures
                if isinstance(data, list): pages_in_batch = data
                elif isinstance(data, dict) and 'trpPage' in data and isinstance(data.get('trpPage'), list): pages_in_batch = data['trpPage']
                elif isinstance(data, dict) and 'pageList' in data and isinstance(data.get('pageList'), dict) and isinstance(data['pageList'].get('pages'), list): pages_in_batch = data['pageList']['pages']
                elif isinstance(data, dict) and 'pages' in data and isinstance(data.get('pages'), list): pages_in_batch = data['pages']
                else: logger.warning(f"Unexpected JSON structure in page list response for doc {doc_id}: {data if isinstance(data, dict) else type(data)}"); break

                if not pages_in_batch:
                    if offset == 0: logger.debug(f"No pages found for document {doc_id}.")
                    break

                for page_info in pages_in_batch:
                    if isinstance(page_info, dict) and 'pageNr' in page_info and 'pageId' in page_info:
                        try:
                            all_pages.append({
                                'pageNr': int(page_info['pageNr']),
                                'pageId': int(page_info['pageId']),
                                'imgFileName': page_info.get('imgFileName') # Include filename if present
                            })
                        except (ValueError, TypeError): logger.warning(f"Skipping page due to invalid pageNr/pageId format: {page_info}")
                    else: logger.warning(f"Skipping page entry with missing fields: {page_info}")

                logger.debug(f"Fetched {len(pages_in_batch)} pages for doc {doc_id} (Total: {len(all_pages)}).")

                if len(pages_in_batch) < limit: break
                offset += limit

            except json.JSONDecodeError as e:
                logger.error(f"Could not decode JSON response for page list: {e}")
                logger.error(f"Response text: {response.text[:500]}")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing page list chunk: {e}", exc_info=True)
                break

        logger.debug(f"Finished fetching pages for Doc {doc_id}. Total retrieved: {len(all_pages)}")
        all_pages.sort(key=lambda p: p['pageNr']) # Ensure sorted
        return all_pages

    def get_page_details(self, coll_id: int, doc_id: int, page_index: int = 0) -> Optional[Tuple[int, int, str]]:
        """Gets details (pageNr, pageId, imgUrl) for a specific page index."""
        logger.info(f"Getting page details for Doc {doc_id}, Page Index {page_index}...")
        endpoint = f"collections/{coll_id}/{doc_id}/pages"
        params = {"nValues": 1, "index": page_index}
        response = self._make_request("GET", endpoint, params=params, timeout=30)

        if response is None: return None, None, None

        try:
            response_data = response.json()
            page_list = []
            if isinstance(response_data, dict) and 'trpPage' in response_data and isinstance(response_data['trpPage'], list): page_list = response_data['trpPage']
            elif isinstance(response_data, list): page_list = response_data
            else: logger.error(f"Unexpected structure in get pages response: {response_data}"); return None, None, None

            if page_list:
                page_info = page_list[0]
                page_nr = page_info.get('pageNr')
                page_id = page_info.get('pageId')
                img_url = page_info.get('url') or page_info.get('imageUrl')
                if page_nr is not None and page_id is not None and img_url:
                    try: return int(page_nr), int(page_id), img_url
                    except (ValueError, TypeError): logger.error(f"Could not convert pageNr/pageId to int: {page_nr}, {page_id}"); return None, None, None
                else: logger.error(f"Missing required details in page data: {page_info}"); return None, None, None
            else: logger.error(f"Could not find page data at index {page_index} for doc {doc_id}."); return None, None, None
        except json.JSONDecodeError as e:
            logger.error(f"Could not decode JSON response getting page details: {e}"); logger.error(f"Response text: {response.text[:500]}"); return None, None, None
        except Exception as e:
            logger.error(f"Unexpected error getting page details: {e}", exc_info=True); return None, None, None

    def get_page_xml_content(self, coll_id: int, doc_id: int, page_nr: int) -> Optional[bytes]:
        """Downloads the PAGE XML content for a specific page as bytes."""
        logger.debug(f"Downloading PAGE XML content for Doc {doc_id}, Page {page_nr}...")
        endpoint = f"collections/{coll_id}/{doc_id}/{page_nr}/text"
        response = self._make_request("GET", endpoint, timeout=45)

        if response is None: return None

        xml_content = response.content
        if not xml_content or not xml_content.strip().startswith(b'<?xml'):
            logger.warning(f"Downloaded content for Doc {doc_id}, Page {page_nr} not valid XML. Starts with: {xml_content[:100]}")
            return None
        logger.debug(f"PAGE XML content downloaded successfully ({len(xml_content)} bytes).")
        return xml_content

    def download_page_xml_file(self, coll_id: int, doc_id: int, page_nr: int, output_path: str) -> bool:
        """Downloads the PAGE XML for a specific page and saves it to a file."""
        xml_content = self.get_page_xml_content(coll_id, doc_id, page_nr)
        if xml_content:
            try:
                with open(output_path, 'wb') as f:
                    f.write(xml_content)
                logger.info(f"PAGE XML downloaded successfully to {output_path}")
                return True
            except IOError as e:
                logger.error(f"Failed to write downloaded XML to {output_path}: {e}")
                return False
        else:
            return False

    def upload_page_xml(self, coll_id: int, doc_id: int, page_nr: int, xml_bytes: bytes, params: Dict[str, Any]) -> bool:
        """
        Uploads PAGE XML content to a specific page.
        Can be used to overwrite or create new versions depending on params.

        Args:
            coll_id: Collection ID.
            doc_id: Document ID.
            page_nr: Page number.
            xml_bytes: The PAGE XML content as bytes.
            params: Dictionary of query parameters (e.g., status, note, toolName, parent).
        """
        endpoint = f"collections/{coll_id}/{doc_id}/{page_nr}/text"
        headers = {'Content-Type': 'application/xml;charset=UTF-8'}
        logger.info(f"Attempting to POST PAGE XML to {endpoint} with params: {params}")

        # Save XML for debugging just before POST
        debug_xml_dir = "debug_xml_uploads"
        os.makedirs(debug_xml_dir, exist_ok=True)
        safe_target_title = f"doc_{doc_id}"
        debug_xml_filename = os.path.join(debug_xml_dir, f"{safe_target_title}_page_{page_nr}_upload_attempt.xml")
        try:
            with open(debug_xml_filename, "wb") as f_debug: f_debug.write(xml_bytes)
            logger.info(f"Saved XML for upload inspection to: {debug_xml_filename}")
        except Exception as save_err: logger.error(f"Failed to save debug XML file {debug_xml_filename}: {save_err}")


        response = self._make_request("POST", endpoint, headers=headers, params=params, data=xml_bytes, timeout=90)

        if response is not None and response.status_code in [200, 201, 204]:
            logger.info(f"Successfully uploaded/updated page {page_nr} (Doc {doc_id}). Status: {response.status_code}")
            try:
                if os.path.exists(debug_xml_filename): os.remove(debug_xml_filename)
            except OSError: pass
            return True
        else:
            logger.error(f"Failed upload/update for page {page_nr} (Doc {doc_id}). Status: {response.status_code if response else 'N/A'}")
            if response is not None and response.status_code == 500:
                 logger.error(f"--> 500 error on POST. This likely means page {page_nr} already exists and cannot be overwritten via POST without correct 'parent' param, OR the XML is invalid. INSPECT: {debug_xml_filename}")
            return False

    def find_document_by_title(self, coll_id: int, title: str) -> Optional[int]:
        """Finds a document ID by exact title, selecting earliest if multiple."""
        logger.info(f"Searching for document title '{title}' in Collection {coll_id}...")
        endpoint = "collections/findDocuments"
        params = {'collId': coll_id, 'title': title, 'exactMatch': True, 'nValues': 10}
        response = self._make_request("GET", endpoint, params=params, timeout=30)

        if response is None: return None

        try:
            results = response.json()
            doc_list = []
            # Handle variations
            if isinstance(results, list): doc_list = results
            elif isinstance(results, dict) and 'content' in results and isinstance(results['content'], list): doc_list = results['content']
            elif isinstance(results, dict) and 'documents' in results and isinstance(results['documents'], list): doc_list = results['documents']
            else: logger.warning(f"Unexpected structure finding documents: {results}"); doc_list = []

            matching_docs = [doc for doc in doc_list if isinstance(doc, dict) and doc.get('title') == title]

            if len(matching_docs) == 1:
                doc_id = matching_docs[0].get('docId')
                if doc_id: return int(doc_id)
                else: logger.error(f"Found doc matching title, but no docId: {matching_docs[0]}"); return None
            elif len(matching_docs) == 0:
                logger.info(f"Document '{title}' not found in collection {coll_id}.")
                return None
            else:
                logger.warning(f"Found {len(matching_docs)} docs with title '{title}'. Selecting earliest.")
                valid_docs = [d for d in matching_docs if d.get('docId') and d.get('uploadTimestamp')]
                if not valid_docs: logger.error("Multiple matches found, but none have required ID and timestamp."); return None
                try:
                    earliest_doc = min(valid_docs, key=lambda d: d['uploadTimestamp'])
                    return int(earliest_doc['docId'])
                except (TypeError, ValueError, KeyError) as e: logger.error(f"Error selecting earliest doc: {e}"); return None

        except json.JSONDecodeError as e:
            logger.error(f"Could not decode JSON finding document: {e}"); logger.error(f"Response text: {response.text[:500]}"); return None
        except Exception as e:
            logger.error(f"Unexpected error finding document: {e}", exc_info=True); return None

    def get_transcript_list(self, coll_id: int, doc_id: int, page_nr: int) -> Optional[List[Dict[str, Any]]]:
        """Fetches the list of transcript metadata for a specific page."""
        logger.debug(f"Fetching transcript list for Doc {doc_id}, Page {page_nr}")
        endpoint = f"collections/{coll_id}/{doc_id}/{page_nr}/list"
        response = self._make_request("GET", endpoint, timeout=45)

        if response is None: return None

        try:
            transcript_list = response.json()
            if not isinstance(transcript_list, list):
                logger.warning(f"Unexpected response format from transcript list endpoint for Doc {doc_id}, Page {page_nr}. Expected list, got {type(transcript_list)}.")
                return None
            logger.debug(f"Successfully fetched {len(transcript_list)} transcript entries.")
            return transcript_list
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from transcript list: {e}"); logger.error(f"Response text: {response.text[:500]}"); return None
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript list: {e}", exc_info=True); return None

    def download_transcript_by_fileid(self, file_id: str, output_path: str) -> bool:
        """Downloads PAGE XML using the specific file content ID."""
        # Note: This uses a different base URL
        file_get_url = f"https://files.transkribus.eu/Get?id={file_id}"
        logger.debug(f"Attempting download using file ID '{file_id}' from URL: {file_get_url}")
        response = None
        try:
            # Use the same authenticated session
            response = self.session.get(file_get_url, timeout=60)
            response.raise_for_status()

            if not response.content or not response.content.strip().startswith(b'<?xml'):
                logger.error(f"Downloaded content for file ID {file_id} not valid XML. Status: {response.status_code}. Starts with: {response.content[:150]}")
                if response.status_code >= 400: logger.error(f"Response text: {response.text[:500]}")
                return False

            with open(output_path, 'wb') as f: f.write(response.content)
            logger.debug(f"PAGE XML for file ID {file_id} downloaded successfully to {output_path}.")
            return True
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 'N/A'
            logger.error(f"HTTP error {status} downloading file ID {file_id}: {e}")
            if e.response is not None: logger.error(f"Response: {e.response.text[:500]}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Network/HTTP error downloading file ID {file_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading file ID {file_id}: {e}", exc_info=True)
            return False

    def delete_document(self, coll_id: int, doc_id: int, dry_run: bool = False) -> bool:
        """Deletes a document from the collection."""
        endpoint = f"collections/{coll_id}/{doc_id}/delete"
        if dry_run:
            logger.info(f"[DRY RUN] Would delete document ID {doc_id} from Collection {coll_id}")
            return True
        logger.warning(f"Attempting to DELETE document ID {doc_id} from Collection {coll_id}")
        response = self._make_request("DELETE", endpoint, timeout=45)
        if response is not None and response.status_code in [200, 204]:
            logger.info(f"Successfully DELETED document ID {doc_id}. Status: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to delete document ID {doc_id}. Status: {response.status_code if response else 'N/A'}")
            return False

    def delete_page(self, coll_id: int, doc_id: int, page_nr: int, dry_run: bool = False) -> bool:
        """Deletes a specific page from a document."""
        endpoint = f"collections/{coll_id}/{doc_id}/{page_nr}"
        if dry_run:
            logger.info(f"[DRY RUN] Would delete Page {page_nr} from Doc {doc_id}")
            return True
        logger.warning(f"Attempting to DELETE Page {page_nr} from Doc {doc_id}")
        response = self._make_request("DELETE", endpoint, timeout=30)
        if response is not None and response.status_code in [200, 204]:
            logger.info(f"Successfully DELETED Page {page_nr} from Doc {doc_id}. Status: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to delete Page {page_nr} from Doc {doc_id}. Status: {response.status_code if response else 'N/A'}")
            return False

    def remove_doc_from_collection(self, coll_id: int, doc_id: int, dry_run: bool = False) -> bool:
        """Removes a document from a specific collection (doesn't delete permanently)."""
        endpoint = f"collections/{coll_id}/removeDocFromCollection"
        params = {'id': doc_id}
        if dry_run:
            logger.info(f"[DRY RUN] Would remove Doc ID {doc_id} from Collection {coll_id}")
            return True
        logger.warning(f"Attempting to REMOVE Doc ID {doc_id} from Collection {coll_id}")
        response = self._make_request("POST", endpoint, params=params, timeout=30)
        if response is not None and response.status_code in [200, 204]:
            logger.info(f"Successfully REMOVED Doc ID {doc_id} from Collection {coll_id}. Status: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to remove Doc ID {doc_id} from Collection {coll_id}. Status: {response.status_code if response else 'N/A'}")
            return False

    def check_page_xml_exists(self, coll_id: int, doc_id: int, page_nr: int) -> Optional[bool]:
        """Checks if PAGE XML exists for a specific page using a HEAD request."""
        logger.debug(f"Checking existence of PAGE XML for Doc {doc_id}, Page {page_nr}...")
        endpoint = f"collections/{coll_id}/{doc_id}/{page_nr}/text"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.head(url, timeout=15)
            if response.status_code == 200:
                logger.debug(f"PAGE XML exists for Doc {doc_id}, Page {page_nr} (Status: 200)")
                return True
            elif response.status_code == 404:
                logger.debug(f"PAGE XML does NOT exist for Doc {doc_id}, Page {page_nr} (Status: 404)")
                return False
            else:
                logger.warning(f"Unexpected status {response.status_code} checking PAGE XML for Doc {doc_id}, Page {page_nr}. Assuming it might exist.")
                return True # Be conservative
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout checking PAGE XML for Doc {doc_id}, Page {page_nr}. Assuming it might exist.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking PAGE XML for Doc {doc_id}, Page {page_nr}: {e}")
            logger.warning("Assuming PAGE XML might exist due to network error.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error checking PAGE XML for Doc {doc_id}, Page {page_nr}: {e}", exc_info=True)
            logger.warning("Assuming PAGE XML might exist due to unexpected error.")
            return True

    # --- Upload New Document Methods ---
    def _create_upload_structure(self, coll_id: int, title: str, page_filename: str) -> Optional[int]:
        """Step 1 of upload: Create the document structure."""
        endpoint = "uploads"
        params = {'collId': coll_id}
        payload = {
            "md": {"title": title},
            "pageList": {"pages": [{"fileName": page_filename, "pageNr": 1}]}
        }
        response = self._make_request("POST", endpoint, params=params, json=payload, timeout=30)
        if response is None: return None

        try:
            xml_root = ET.fromstring(response.text)
            upload_id_elem = xml_root.find('uploadId')
            if upload_id_elem is not None and upload_id_elem.text:
                upload_id = int(upload_id_elem.text)
                logger.debug(f"Successfully created upload structure for '{title}'. Upload ID: {upload_id}")
                return upload_id
            else:
                logger.error(f"Failed to find 'uploadId' in response for '{title}'. Response: {response.text[:500]}")
                return None
        except ET.ParseError as xml_err:
            logger.error(f"Could not parse XML response for '{title}': {xml_err}. Response: {response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing upload structure response for '{title}': {e}", exc_info=True)
            return None

    def _upload_page_file(self, upload_id: int, local_path: str, filename_in_trp: str) -> bool:
        """Step 2 of upload: Upload the actual image file."""
        endpoint = f"uploads/{upload_id}"
        try:
            with open(local_path, 'rb') as f:
                files = {'img': (filename_in_trp, f, 'application/octet-stream')} # Use filename_in_trp here
                response = self._make_request("PUT", endpoint, files=files, timeout=180) # Longer timeout for upload
            if response is not None and response.status_code in [200, 201, 204]:
                logger.info(f"File upload request successful for '{filename_in_trp}' (Upload ID: {upload_id}). Status: {response.status_code}")
                return True
            else:
                logger.error(f"File upload failed for '{filename_in_trp}'. Status: {response.status_code if response else 'N/A'}")
                return False
        except FileNotFoundError:
            logger.error(f"Local file not found for upload: {local_path}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during file upload of '{filename_in_trp}': {e}", exc_info=True)
            return False

    def upload_new_document(self, coll_id: int, local_path: str, filename_in_trp: str) -> bool:
        """Uploads an image as a new document in two steps."""
        logger.info(f"Starting 2-step upload for {filename_in_trp} to Collection {coll_id}...")
        upload_id = self._create_upload_structure(coll_id, filename_in_trp, filename_in_trp)
        if upload_id is None:
            logger.error("Failed to create upload structure. Aborting upload.")
            return False

        upload_success = self._upload_page_file(upload_id, local_path, filename_in_trp)
        if upload_success:
            logger.info(f"Successfully completed 2-step upload for {filename_in_trp}.")
            # Note: We don't poll for completion here, assuming it processes asynchronously.
            return True
        else:
            logger.error(f"File upload step failed for {filename_in_trp}.")
            # Consider deleting the created structure if the file upload fails? (API might not support this easily)
            return False

    def trigger_pylaia_htr(self, coll_id: int, model_id: int, doc_id: int, page_nr: int) -> bool:
        """Triggers HTR recognition using a PyLaia model for a specific page."""
        endpoint = f"pylaia/{coll_id}/{model_id}/recognition"
        if page_nr is None or not isinstance(page_nr, int) or page_nr < 0:
            logger.error(f"Invalid Page Number ({page_nr}) for HTR trigger.")
            return False

        payload = {"docs": [{"docId": doc_id, "pageNrs": [page_nr]}]}
        logger.info(f"Triggering PyLaia HTR (Model ID: {model_id}) for Doc {doc_id}, Page {page_nr}...")
        logger.debug(f"HTR Payload: {json.dumps(payload)}")
        response = self._make_request("POST", endpoint, json=payload, timeout=45)

        if response is not None and response.status_code in [200, 201, 202]:
            logger.info("HTR job submitted successfully.")
            try: logger.info(f"Job Info: {json.dumps(response.json(), indent=2)}")
            except json.JSONDecodeError: logger.warning("HTR response was not JSON, but status was ok.")
            return True
        else:
            logger.error(f"HTR trigger failed. Status: {response.status_code if response else 'N/A'}")
            return False