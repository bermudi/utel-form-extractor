import os
import csv
import argparse
import dotenv
import json
import base64
import logging
import dotenv
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
from google import genai
from openai import OpenAI

<<<<<<< HEAD
# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging for generous console output
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logging; change to INFO for less verbosity.
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
=======
dotenv.load_dotenv()


#########################
# TODO: make sure response doesn't contain any new lines that will affect the csv
#########################
>>>>>>> master

# Custom schema matching the UTEL University feedback form
class UTELFormData(BaseModel):
    """Schema for UTEL University feedback form data extraction"""
    nombre: Optional[str] = Field(None, description="Nombre del estudiante")
    matricula: Optional[str] = Field(None, description="Número de matrícula")
    correo_electronico: Optional[str] = Field(None, description="Correo electrónico del estudiante")
    calificacion_evento: Optional[str] = Field(None, description="Calificación del evento (Excelente, Buena, Aceptable, Mejorable)")
    comentario_adicional: Optional[str] = Field(None, description="Sugerencia o comentario adicional sobre el evento")
    actividad_tiempo_libre: Optional[str] = Field(None, description="Actividad que disfruta en tiempo libre")
    area_apoyo: Optional[str] = Field(None, description="Área en la que requiere apoyo (Académico, Administrativo, Aula Virtual, Comunicación, Otro)")
    detalles_situacion: Optional[str] = Field(None, description="Detalles adicionales sobre la situación")

def setup_gemini_client():
    """Set up and return the Gemini API client"""
    logging.info("Setting up Gemini API client.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        error_msg = ("GEMINI_API_KEY environment variable not set. "
                     "Please set it with your API key from Google AI Studio.")
        logging.error(error_msg)
        raise ValueError(error_msg)
    logging.debug("Gemini API key found. Initializing client.")
    return genai.Client(api_key=api_key)

def setup_llama_client():
    """Set up and return the OpenAI client for OpenRouter/Llama"""
    logging.info("Setting up Llama client using OPENROUTER_API_KEY.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        error_msg = ("OPENROUTER_API_KEY environment variable not set. "
                     "Please set it with your API key from OpenRouter.")
        logging.error(error_msg)
        raise ValueError(error_msg)
    logging.debug("Llama API key found. Initializing client.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def process_with_gemini(client, image_path, timeout=60, max_retries=2):
    """Process a UTEL University feedback form with Gemini"""
<<<<<<< HEAD
    logging.info(f"Processing image with Gemini: {image_path}")
    try:
        # Read the image file
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        logging.debug(f"Read {len(image_bytes)} bytes from {image_path}.")
        
        # Determine mime type based on file extension
        ext = Path(image_path).suffix.lower()
        mime_type = "image/jpeg"  # Default
        if ext == ".png":
            mime_type = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
        logging.info(f"Determined mime type for {image_path}: {mime_type}")
        
        # Create specific prompt for UTEL form extraction
        prompt = """
        Extract the following information from this UTEL University feedback form:
        1. Nombre (Name): Look for "Nombre:" field and extract the handwritten name
        2. Matrícula (Student ID): Look for "Matrícula:" field and extract the ID number
        3. Correo electrónico (Email): Look for "Correo electrónico:" field and extract the email address
        4. Calificación evento: Identify which option is marked with X (Excelente, Buena, Aceptable, or Mejorable)
        5. Comentario adicional: Extract any handwritten comment in the suggestions section
        6. Actividad tiempo libre: Extract the response to the question about hobbies/free time activities
        7. Área apoyo: Identify which support area is marked with X (Académico, Administrativo, Aula Virtual, Comunicación, or Otro)
        8. Detalles situación: Extract the handwritten details about their situation
        
        Return the extracted data in structured format according to the schema.
        """
        logging.debug("Generated prompt for Gemini model.")
        
        # Generate content with structured output
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': UTELFormData,
            }
        )
        logging.info("Gemini processing complete, response received.")
        return response.parsed
            
    except Exception as e:
        logging.error(f"Error processing with Gemini for {image_path}: {e}")
        raise

def process_with_llama(client, image_path, site_url="https://example.com", site_name="UTEL Form Extractor"):
    """Process a UTEL University feedback form with Llama 3.2 via OpenRouter"""
    logging.info(f"Processing image with Llama: {image_path}")
    try:
        # Convert image to base64
        with open(image_path, "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        logging.debug(f"Encoded image {image_path} to base64.")
        
        # Determine mime type based on file extension
        ext = Path(image_path).suffix.lower()
        mime_type = "image/jpeg"  # Default
        if ext == ".png":
            mime_type = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
        logging.info(f"Determined mime type for {image_path}: {mime_type}")
            
        data_url = f"data:{mime_type};base64,{base64_image}"
        logging.debug("Generated data URL for image.")
        
        prompt = """
        Extract information from this UTEL University feedback form.
        
        Please extract the following fields and provide them in a JSON format:
        - nombre: The student's name from the "Nombre:" field
        - matricula: The student ID from the "Matrícula:" field
        - correo_electronico: The email from the "Correo electrónico:" field
        - calificacion_evento: Which option is marked with X (Excelente, Buena, Aceptable, or Mejorable)
        - comentario_adicional: Any handwritten comment in the suggestions section
        - actividad_tiempo_libre: The response about hobbies/free time activities
        - area_apoyo: Which support area is marked with X (Académico, Administrativo, Aula Virtual, Comunicación, or Otro)
        - detalles_situacion: The handwritten details about their situation
        
        IMPORTANT: Return ONLY a valid JSON object with these fields, nothing else.
        """
        logging.debug("Generated prompt for Llama model.")
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            model="meta-llama/llama-3.2-11b-vision-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        logging.info("Llama processing complete, response received.")
        
        # Parse the JSON response
        response_text = completion.choices[0].message.content
        logging.debug(f"Llama response text: {response_text}")
        response_data = json.loads(response_text)
        logging.info("Parsed Llama response into JSON successfully.")
        
        # Convert to Pydantic model
        return UTELFormData(**response_data)
            
    except Exception as e:
        logging.error(f"Error processing with Llama for {image_path}: {e}")
        raise
=======
    for retry in range(max_retries + 1):
        try:
            if retry > 0:
                print(f"Retry {retry}/{max_retries} for Gemini processing of {image_path}")
                
            # Read the image file
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Determine mime type based on file extension
            ext = Path(image_path).suffix.lower()
            mime_type = "image/jpeg"  # Default
            if ext == ".png":
                mime_type = "image/png"
            elif ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".webp":
                mime_type = "image/webp"
            
            # Create specific prompt for UTEL form extraction
            prompt = """
            Extract the following information from this UTEL University feedback form:
            1. Nombre (Name): Look for "Nombre:" field and extract the handwritten name
            2. Matrícula (Student ID): Look for "Matrícula:" field and extract the ID number
            3. Correo electrónico (Email): Look for "Correo electrónico:" field and extract the email address
            4. Calificación evento: Identify which option is marked with X (Excelente, Buena, Aceptable, or Mejorable)
            5. Comentario adicional: Extract any handwritten comment in the suggestions section
            6. Actividad tiempo libre: Extract the response to the question about hobbies/free time activities
            7. Área apoyo: Identify which support area is marked with X (Académico, Administrativo, Aula Virtual, Comunicación, or Otro)
            8. Detalles situación: Extract the handwritten details about their situation
            
            Return the extracted data in structured format according to the schema.
            """
            
            try:
                # Generate content with structured output with timeout
                import concurrent.futures
                import time
                
                def generate_with_timeout():
                    return client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[
                            prompt,
                            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                        ],
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': UTELFormData,
                        }
                    )
                
                # Use ThreadPoolExecutor to implement timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(generate_with_timeout)
                    try:
                        response = future.result(timeout=timeout)  # Use provided timeout
                        if response and hasattr(response, 'parsed'):
                            result = response.parsed
                            # Log the extracted data for debugging
                            if hasattr(result, 'model_dump'):
                                print(f"Gemini extracted data: {json.dumps(result.model_dump(), indent=2)[:500]}...")
                            elif hasattr(result, 'dict'):
                                print(f"Gemini extracted data: {json.dumps(result.dict(), indent=2)[:500]}...")
                            return result
                        else:
                            print(f"Warning: Empty or invalid response from Gemini for {image_path}")
                            if retry < max_retries:
                                continue  # Try again
                            return UTELFormData()
                    except concurrent.futures.TimeoutError:
                        print(f"Timeout error processing with Gemini for {image_path}")
                        if retry < max_retries:
                            continue  # Try again
                        return UTELFormData()
                    
            except Exception as e:
                print(f"Error in Gemini API call for {image_path}: {e}")
                if retry < max_retries:
                    continue  # Try again
                return UTELFormData()
                
        except Exception as e:
            print(f"Error processing with Gemini {image_path}: {e}")
            if retry < max_retries:
                continue  # Try again
            return UTELFormData()  # Return empty model instead of raising exception
    
    # If we get here, all retries failed
    return UTELFormData()

def process_with_llama(client, image_path, site_url="https://example.com", site_name="UTEL Form Extractor", timeout=60, max_retries=2):
    """Process a UTEL University feedback form with Llama 3.2 via OpenRouter"""
    for retry in range(max_retries + 1):
        try:
            if retry > 0:
                print(f"Retry {retry}/{max_retries} for Llama processing of {image_path}")
                
            # Convert image to base64
            with open(image_path, "rb") as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine mime type based on file extension
            ext = Path(image_path).suffix.lower()
            mime_type = "image/jpeg"  # Default
            if ext == ".png":
                mime_type = "image/png"
            elif ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".webp":
                mime_type = "image/webp"
                
            data_url = f"data:{mime_type};base64,{base64_image}"
            
            prompt = """
            Extract the following information from this UTEL University feedback form image. For each field, carefully look at the form and extract the exact text or marked option:

            1. For "Nombre:" field - Look for the handwritten name and extract it exactly as written
            2. For "Matrícula:" field - Find and extract the student ID number
            3. For "Correo electrónico:" field - Extract the complete email address
            4. For "Calificación del evento:" section - Find which option (Excelente, Buena, Aceptable, or Mejorable) is marked with an X
            5. For the suggestions section - Extract any handwritten comment word for word
            6. For the free time activities question - Copy the exact response about what activities they enjoy
            7. For "Área de apoyo:" section - Identify which option (Académico, Administrativo, Aula Virtual, Comunicación, or Otro) is marked
            8. For the situation details - Copy the complete handwritten explanation

            Format your response as a JSON object with these exact fields:
            {
                "nombre": "extracted name",
                "matricula": "extracted ID",
                "correo_electronico": "extracted email",
                "calificacion_evento": "marked option",
                "comentario_adicional": "extracted comment",
                "actividad_tiempo_libre": "extracted activities",
                "area_apoyo": "marked area",
                "detalles_situacion": "extracted details"
            }

            If you cannot read or determine a field clearly, use null for that field. Return ONLY the JSON object, no other text.
            """
            
            try:
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": site_url,
                        "X-Title": site_name,
                    },
                    model="meta-llama/llama-3.2-11b-vision-instruct:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        }
                    ],
                    response_format={"type": "json_object"},
                    timeout=120  # Increased timeout to 120 seconds
                )
                
                # Check if completion and choices exist
                if not completion or not completion.choices or not completion.choices[0].message.content:
                    print(f"Warning: Empty or invalid response from Llama for {image_path}")
                    if retry < max_retries:
                        continue  # Try again
                    return UTELFormData()
                
                # Parse the JSON response
                response_text = completion.choices[0].message.content
                
                try:
                    response_data = json.loads(response_text)
                    # Log the extracted data for debugging
                    print(f"Llama extracted data: {json.dumps(response_data, indent=2)[:500]}...")
                    # Convert to Pydantic model
                    return UTELFormData(**response_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from Llama response for {image_path}: {e}")
                    print(f"Raw response: {response_text[:500]}...")  # Print first 500 chars of response
                    if retry < max_retries:
                        continue  # Try again
                    return UTELFormData()
                    
            except Exception as e:
                print(f"Error in OpenAI API call for {image_path}: {e}")
                if retry < max_retries:
                    continue  # Try again
                return UTELFormData()
                
        except Exception as e:
            print(f"Error processing with Llama {image_path}: {e}")
            if retry < max_retries:
                continue  # Try again
            return UTELFormData()  # Return empty model instead of raising exception
    
    # If we get here, all retries failed
    return UTELFormData()
>>>>>>> master

def compare_results(gemini_data, llama_data):
    """
    Compare results from both models and identify discrepancies
    
    Returns:
        Tuple of (merged_data, has_discrepancies, discrepancies_dict)
    """
<<<<<<< HEAD
    logging.info("Comparing results from Gemini and Llama.")
=======
    # Handle cases where one model might have failed
    if gemini_data is None:
        print("Warning: Gemini data is None, using empty model")
        gemini_data = UTELFormData()
        
    if llama_data is None:
        print("Warning: Llama data is None, using empty model")
        llama_data = UTELFormData()
    
>>>>>>> master
    # Convert to dictionaries
    if hasattr(gemini_data, 'model_dump'):
        gemini_dict = gemini_data.model_dump()
    elif hasattr(gemini_data, 'dict'):
        gemini_dict = gemini_data.dict()
    else:
        gemini_dict = gemini_data
        
    if hasattr(llama_data, 'model_dump'):
        llama_dict = llama_data.model_dump()
    elif hasattr(llama_data, 'dict'):
        llama_dict = llama_data.dict()
    else:
        llama_dict = llama_data
    
    discrepancies = {}
    for key in gemini_dict:
        if key in llama_dict:
            if (not gemini_dict[key] and not llama_dict[key]):
                continue
                
<<<<<<< HEAD
            gemini_val = str(gemini_dict[key]).lower().strip() if gemini_dict[key] else ""
            llama_val = str(llama_dict[key]).lower().strip() if llama_dict[key] else ""
            
            if gemini_val != llama_val:
=======
            # Check for discrepancies
            if gemini_dict[key] != llama_dict[key]:
>>>>>>> master
                discrepancies[key] = {
                    "gemini": gemini_dict[key],
                    "llama": llama_dict[key]
                }
<<<<<<< HEAD
                logging.debug(f"Discrepancy found for field '{key}': Gemini='{gemini_dict[key]}', Llama='{llama_dict[key]}'")
    
    merged_data = {}
    for key in gemini_dict:
        if key in discrepancies:
            merged_data[key] = gemini_dict[key]
            merged_data[f"{key}_verified"] = False
        else:
            merged_data[key] = gemini_dict[key]
            merged_data[f"{key}_verified"] = True
=======
                print(f"Discrepancy in field '{key}':")
                print(f"  - Gemini: '{gemini_dict[key]}'")
                print(f"  - Llama:  '{llama_dict[key]}'")
    
    # Merge data, preferring Gemini when there are discrepancies
    merged_data = UTELFormData(**gemini_dict)
>>>>>>> master
    
    # Store discrepancies in the model for later use
    setattr(merged_data, '_discrepancies', discrepancies)
    
    # Add verification fields if there are discrepancies
    has_discrepancies = len(discrepancies) > 0
<<<<<<< HEAD
    if has_discrepancies:
        logging.info(f"Total discrepancies found: {len(discrepancies)}")
    else:
        logging.info("No discrepancies found between Gemini and Llama results.")
=======
    
>>>>>>> master
    return merged_data, has_discrepancies, discrepancies

def save_to_csv(data_list, output_file, include_verification=True, append=False):
    """Save extracted form data to a CSV file"""
    logging.info(f"Saving extracted data to CSV file: {output_file}")
    if not data_list:
<<<<<<< HEAD
        logging.warning("No data to save to CSV.")
        print("No data to save.")
=======
        print("No data to save to CSV.")
>>>>>>> master
        return
        
    # Determine fields based on the first item
    first_item = data_list[0]
    if hasattr(first_item, 'model_dump'):
        fields = list(first_item.model_dump().keys())
    elif hasattr(first_item, 'dict'):
        fields = list(first_item.dict().keys())
    else:
        fields = list(first_item.keys())
    
<<<<<<< HEAD
    field_names = list(data_list[0].keys())
    
    if not include_verification:
        field_names = [field for field in field_names if not field.endswith("_verified")]
        data_list = [{k: v for k, v in item.items() if not k.endswith("_verified")} 
                     for item in data_list]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data_list)
        logging.info(f"Data successfully saved to {output_file}")
        logging.info(f"Total forms processed: {len(data_list)}")
    except Exception as e:
        logging.error(f"Error saving CSV file {output_file}: {e}")
        raise
=======
    # Add verification fields if requested
    all_fields = fields.copy()
    if include_verification:
        for field in fields:
            all_fields.append(f"{field}_verified")
    
    # Check if file exists and we're appending
    file_exists = os.path.isfile(output_file) and append
>>>>>>> master

    # Write to CSV
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        
        # Only write header if creating a new file
        if not file_exists:
            writer.writeheader()
        
        for item in data_list:
            # Convert to dict if needed
            if hasattr(item, 'model_dump'):
                row = item.model_dump()
            elif hasattr(item, 'dict'):
                row = item.dict()
            else:
                row = item
                
            # Add verification fields if requested
            if include_verification:
                # Get verification status from the discrepancies dict
                # We assume all fields are verified unless they're in the discrepancies
                for field in fields:
                    row[f"{field}_verified"] = True  # Default to verified
                
                # Mark fields with discrepancies as unverified
                if hasattr(item, '_discrepancies') and item._discrepancies:
                    for field in item._discrepancies:
                        if field in fields:
                            row[f"{field}_verified"] = False
            
            writer.writerow(row)
    
    if not append:
        print(f"Data saved to {output_file}")
    else:
        print(f"Data appended to {output_file}")

def save_discrepancies(discrepancies_list, output_file, append=False):
    """Save discrepancies to a JSON file for review"""
<<<<<<< HEAD
    logging.info(f"Saving discrepancies to JSON file: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(discrepancies_list, f, ensure_ascii=False, indent=2)
        logging.info(f"Discrepancies successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving discrepancies to {output_file}: {e}")
        raise
=======
    # If appending, read existing discrepancies first
    existing_discrepancies = []
    if append and os.path.isfile(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_discrepancies = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not read existing discrepancies from {output_file}, creating new file")
            append = False
    
    # Combine existing and new discrepancies
    if append:
        all_discrepancies = existing_discrepancies + discrepancies_list
    else:
        all_discrepancies = discrepancies_list
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_discrepancies, f, ensure_ascii=False, indent=2)
    
    if append:
        print(f"Discrepancies appended to {output_file}")
    else:
        print(f"Discrepancies saved to {output_file}")

def load_checkpoint(checkpoint_file):
    """Load checkpoint from file"""
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not read checkpoint from {checkpoint_file}, starting from scratch")
        return None

def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save checkpoint to file"""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
>>>>>>> master

def main():
    """Main function to run the UTEL form processing script"""
    logging.info("Starting UTEL form processing script.")
    parser = argparse.ArgumentParser(description='Extract data from UTEL University feedback forms using multiple LLMs')
    parser.add_argument('--input', required=True, help='Path to a form image or directory of form images')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    parser.add_argument('--site-url', default="https://example.com", help='Your site URL for OpenRouter attribution')
    parser.add_argument('--site-name', default="UTEL Form Extractor", help='Your site name for OpenRouter attribution')
    parser.add_argument('--verification', action='store_true', help='Include verification fields in CSV output')
    parser.add_argument('--discrepancies', help='Path to save discrepancies JSON file (optional)')
    parser.add_argument('--skip-llama', action='store_true', help='Skip processing with Llama model')
    parser.add_argument('--skip-gemini', action='store_true', help='Skip processing with Gemini model')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for API calls (default: 60)')
    parser.add_argument('--max-retries', type=int, default=2, help='Maximum number of retries for API calls (default: 2)')
    parser.add_argument('--max-forms', type=int, help='Maximum number of forms to process (optional)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-file', default='.checkpoint.json', help='Path to checkpoint file (default: .checkpoint.json)')
    args = parser.parse_args()
    
    logging.debug(f"Arguments received: {args}")
    
    # Setup clients
    gemini_client = None if args.skip_gemini else setup_gemini_client()
    llama_client = None if args.skip_llama else setup_llama_client()
    
    if not gemini_client and not llama_client:
        raise ValueError("At least one model client must be enabled")
    
    input_path = Path(args.input)
<<<<<<< HEAD
=======
    
    # Load checkpoint if resuming
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint(args.checkpoint_file)
    
    # Process single image or directory of images
>>>>>>> master
    results_list = []
    discrepancies_list = []
    processed_files = []
    
    if input_path.is_file():
        logging.info(f"Processing single form: {input_path}")
        try:
<<<<<<< HEAD
            gemini_data = process_with_gemini(gemini_client, input_path)
            logging.info("✓ Gemini processing complete for the form.")
            llama_data = process_with_llama(llama_client, input_path, args.site_url, args.site_name)
            logging.info("✓ Llama processing complete for the form.")
=======
            # Process with both models
            gemini_data = None
            llama_data = None
            
            if gemini_client:
                try:
                    gemini_data = process_with_gemini(gemini_client, input_path, timeout=args.timeout, max_retries=args.max_retries)
                    print("✓ Gemini processing complete")
                except Exception as e:
                    print(f"Error processing with Gemini {input_path}: {e}")
                    gemini_data = UTELFormData()
            
            if llama_client:
                try:
                    llama_data = process_with_llama(llama_client, input_path, args.site_url, args.site_name, timeout=args.timeout, max_retries=args.max_retries)
                    print("✓ Llama processing complete")
                except Exception as e:
                    print(f"Error processing with Llama {input_path}: {e}")
                    llama_data = UTELFormData()
>>>>>>> master
            
            merged_data, has_discrepancies, discrepancies = compare_results(gemini_data, llama_data)
            results_list.append(merged_data)
            
            if has_discrepancies:
                logging.warning(f"Found discrepancies in form {input_path.name}")
                discrepancies_list.append({
                    "file": str(input_path),
                    "discrepancies": discrepancies
                })
            else:
<<<<<<< HEAD
                logging.info(f"✓ Both models agree on all fields for {input_path.name}")
=======
                print(f"✓ Both models agree on all fields for {input_path.name}")
                
            # Save the extracted data to CSV
            save_to_csv(results_list, args.output, include_verification=args.verification, append=False)
            print(f"\nResults saved to {args.output}")
            
            # Save discrepancies if requested
            if discrepancies_list and args.discrepancies:
                save_discrepancies(discrepancies_list, args.discrepancies, append=False)
            
>>>>>>> master
        except Exception as e:
            logging.error(f"Failed to process {input_path.name}: {str(e)}")
    
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        form_images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
<<<<<<< HEAD
        logging.info(f"Found {len(form_images)} form image(s) to process in directory.")
        for i, image_file in enumerate(form_images):
            logging.info(f"Processing form {i+1}/{len(form_images)}: {image_file.name}")
            try:
                gemini_data = process_with_gemini(gemini_client, image_file)
                logging.info("✓ Gemini processing complete.")
                llama_data = process_with_llama(llama_client, image_file, args.site_url, args.site_name)
                logging.info("✓ Llama processing complete.")
                
                merged_data, has_discrepancies, discrepancies = compare_results(gemini_data, llama_data)
                results_list.append(merged_data)
                
                if has_discrepancies:
                    logging.warning(f"Found discrepancies in form {image_file.name}")
                    discrepancies_list.append({
                        "file": str(image_file),
                        "discrepancies": discrepancies
                    })
                else:
                    logging.info(f"✓ Both models agree on all fields for {image_file.name}")
            except Exception as e:
                logging.error(f"Failed to process {image_file.name}: {str(e)}")
            time.sleep(10)
=======
        # Limit the number of forms if max-forms is specified
        if args.max_forms and args.max_forms > 0 and args.max_forms < len(form_images):
            print(f"Limiting to {args.max_forms} forms (out of {len(form_images)} found)")
            form_images = form_images[:args.max_forms]
        else:
            print(f"Found {len(form_images)} forms to process")
            
        # Resume from checkpoint if available
        if args.resume and checkpoint_data and 'processed_files' in checkpoint_data:
            processed_files = checkpoint_data['processed_files']
            skipped_count = sum(1 for f in form_images if str(f) in processed_files)
            print(f"Resuming from checkpoint, skipping {skipped_count} already processed files")
        
        for i, image_file in enumerate(form_images):
            # Skip already processed files if resuming
            if str(image_file) in processed_files:
                print(f"Skipping already processed file: {image_file.name}")
                continue
                
            print(f"\nProcessing form {i+1}/{len(form_images)}: {image_file.name}")
            try:
                # Process with both models
                gemini_data = None
                llama_data = None
                
                if gemini_client:
                    try:
                        gemini_data = process_with_gemini(gemini_client, image_file, timeout=args.timeout, max_retries=args.max_retries)
                        print("✓ Gemini processing complete")
                    except Exception as e:
                        print(f"Error processing with Gemini {image_file}: {e}")
                        gemini_data = UTELFormData()
                
                if llama_client:
                    try:
                        llama_data = process_with_llama(llama_client, image_file, args.site_url, args.site_name, timeout=args.timeout, max_retries=args.max_retries)
                        print("✓ Llama processing complete")
                    except Exception as e:
                        print(f"Error processing with Llama {image_file}: {e}")
                        llama_data = UTELFormData()
                
                # Compare and merge results only if at least one model succeeded
                if gemini_data or llama_data:
                    merged_data, has_discrepancies, discrepancies = compare_results(gemini_data, llama_data)
                    results_list.append(merged_data)
                    
                    if has_discrepancies:
                        print(f"⚠ Found discrepancies in form {image_file.name}")
                        discrepancies_list.append({
                            "file": str(image_file),
                            "discrepancies": discrepancies
                        })
                    else:
                        print(f"✓ Models agree on all fields for {image_file.name}")
                else:
                    print(f"⚠ Both models failed to process {image_file.name}")
                    
                # Save the extracted data to CSV incrementally
                save_to_csv([merged_data], args.output, include_verification=args.verification, append=True)
                print(f"Data appended to {args.output}")
                
                # Save discrepancies incrementally if requested
                if discrepancies_list and args.discrepancies:
                    save_discrepancies([discrepancies_list[-1]], args.discrepancies, append=True)
                
                # Save checkpoint
                if not processed_files:
                    processed_files = []
                processed_files.append(str(image_file))
                checkpoint_data = {
                    'processed_files': processed_files
                }
                save_checkpoint(args.checkpoint_file, checkpoint_data)
                print(f"Checkpoint saved, {len(processed_files)} files processed so far")
                
            except Exception as e:
                print(f"Failed to process {image_file.name}: {str(e)}")
                continue  # Continue with next image even if this one fails
>>>>>>> master
    
    else:
        error_msg = f"Invalid input path: {input_path}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Print summary
    logging.info("Processing Summary:")
    logging.info(f"Total forms processed: {len(results_list)}")
    logging.info(f"Forms with discrepancies: {len(discrepancies_list)}")
    if discrepancies_list:
        if args.discrepancies:
            logging.info(f"Discrepancies saved to: {args.discrepancies}")
        else:
            logging.info("Use --discrepancies argument to save details about the discrepancies")

if __name__ == "__main__":
    main()