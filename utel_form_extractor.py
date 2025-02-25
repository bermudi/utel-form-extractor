import os
import csv
import argparse
import time
import json
import base64
import logging
import dotenv
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
from google import genai
from openai import OpenAI

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging for generous console output
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logging; change to INFO for less verbosity.
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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

def process_with_gemini(client, image_path):
    """Process a UTEL University feedback form with Gemini"""
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

def compare_results(gemini_data, llama_data):
    """
    Compare results from both models and identify discrepancies
    
    Returns:
        Tuple of (merged_data, has_discrepancies, discrepancies_dict)
    """
    logging.info("Comparing results from Gemini and Llama.")
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
                
            gemini_val = str(gemini_dict[key]).lower().strip() if gemini_dict[key] else ""
            llama_val = str(llama_dict[key]).lower().strip() if llama_dict[key] else ""
            
            if gemini_val != llama_val:
                discrepancies[key] = {
                    "gemini": gemini_dict[key],
                    "llama": llama_dict[key]
                }
                logging.debug(f"Discrepancy found for field '{key}': Gemini='{gemini_dict[key]}', Llama='{llama_dict[key]}'")
    
    merged_data = {}
    for key in gemini_dict:
        if key in discrepancies:
            merged_data[key] = gemini_dict[key]
            merged_data[f"{key}_verified"] = False
        else:
            merged_data[key] = gemini_dict[key]
            merged_data[f"{key}_verified"] = True
    
    has_discrepancies = len(discrepancies) > 0
    if has_discrepancies:
        logging.info(f"Total discrepancies found: {len(discrepancies)}")
    else:
        logging.info("No discrepancies found between Gemini and Llama results.")
    return merged_data, has_discrepancies, discrepancies

def save_to_csv(data_list, output_file, include_verification=True):
    """Save extracted form data to a CSV file"""
    logging.info(f"Saving extracted data to CSV file: {output_file}")
    if not data_list:
        logging.warning("No data to save to CSV.")
        print("No data to save.")
        return
    
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

def save_discrepancies(discrepancies_list, output_file):
    """Save discrepancies to a JSON file for review"""
    logging.info(f"Saving discrepancies to JSON file: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(discrepancies_list, f, ensure_ascii=False, indent=2)
        logging.info(f"Discrepancies successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving discrepancies to {output_file}: {e}")
        raise

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
    args = parser.parse_args()
    
    logging.debug(f"Arguments received: {args}")
    
    # Setup clients
    gemini_client = setup_gemini_client()
    llama_client = setup_llama_client()
    
    input_path = Path(args.input)
    results_list = []
    discrepancies_list = []
    
    if input_path.is_file():
        logging.info(f"Processing single form: {input_path}")
        try:
            gemini_data = process_with_gemini(gemini_client, input_path)
            logging.info("✓ Gemini processing complete for the form.")
            llama_data = process_with_llama(llama_client, input_path, args.site_url, args.site_name)
            logging.info("✓ Llama processing complete for the form.")
            
            merged_data, has_discrepancies, discrepancies = compare_results(gemini_data, llama_data)
            results_list.append(merged_data)
            
            if has_discrepancies:
                logging.warning(f"Found discrepancies in form {input_path.name}")
                discrepancies_list.append({
                    "file": str(input_path),
                    "discrepancies": discrepancies
                })
            else:
                logging.info(f"✓ Both models agree on all fields for {input_path.name}")
        except Exception as e:
            logging.error(f"Failed to process {input_path.name}: {str(e)}")
    
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        form_images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
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
    
    else:
        error_msg = f"Invalid input path: {input_path}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Save the extracted data to CSV
    save_to_csv(results_list, args.output, include_verification=args.verification)
    
    # Save discrepancies if requested
    if discrepancies_list and args.discrepancies:
        save_discrepancies(discrepancies_list, args.discrepancies)
        
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