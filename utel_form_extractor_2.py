import os
import csv
import argparse
import dotenv
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from google import genai

# Load environment variables from .env file
dotenv.load_dotenv()

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

def setup_client():
    """Set up and return the Gemini API client"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with your API key from Google AI Studio."
        )
    return genai.Client(api_key=api_key)

def process_utel_form(client, image_path):
    """
    Process a UTEL University feedback form and extract data
    
    Args:
        client: Configured Gemini API client
        image_path: Path to the form image
        
    Returns:
        Structured data extracted from the form
    """
    try:
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
        return response.parsed
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        raise

def save_to_csv(data_list, output_file):
    """Save extracted form data to a CSV file"""
    if not data_list:
        print("No data to save.")
        return
    
    # Convert data objects to dictionaries
    dict_data = []
    for data in data_list:
        if hasattr(data, 'model_dump'):
            # For newer Pydantic versions (v2+)
            dict_data.append(data.model_dump())
        elif hasattr(data, 'dict'):
            # For older Pydantic versions
            dict_data.append(data.dict())
        else:
            dict_data.append(data)
    
    # Get field names from the first item
    field_names = list(dict_data[0].keys())
    
    # Write the data to the CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(dict_data)
    
    print(f"Data saved to {output_file}")
    print(f"Processed {len(data_list)} forms successfully")

def main():
    """Main function to run the UTEL form processing script"""
    parser = argparse.ArgumentParser(description='Extract data from UTEL University feedback forms using Gemini AI')
    parser.add_argument('--input', required=True, help='Path to a form image or directory of form images')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    args = parser.parse_args()
    
    # Setup the Gemini API client
    client = setup_client()
    
    input_path = Path(args.input)
    
    # Process single image or directory of images
    data_list = []
    
    if input_path.is_file():
        # Process a single form image
        print(f"Processing single form: {input_path}")
        data = process_utel_form(client, input_path)
        data_list.append(data)
    elif input_path.is_dir():
        # Process all form images in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        form_images = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(form_images)} forms to process")
        for i, image_file in enumerate(form_images):
            print(f"Processing form {i+1}/{len(form_images)}: {image_file.name}")
            try:
                data = process_utel_form(client, image_file)
                data_list.append(data)
                print(f"Successfully processed {image_file.name}")
            except Exception as e:
                print(f"Failed to process {image_file.name}: {str(e)}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    # Save the extracted data to CSV
    save_to_csv(data_list, args.output)

if __name__ == "__main__":
    main()  
