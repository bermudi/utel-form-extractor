import os
import json
import base64
import dotenv
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables
dotenv.load_dotenv()

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

def setup_llama_client():
    """Set up and return the OpenAI client for OpenRouter/Llama"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it with your API key from OpenRouter."
        )
    
    # Print API key first few and last few characters for verification
    print(f"API Key (masked): {api_key[:8]}...{api_key[-4:]}")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Test the models endpoint
    try:
        models = client.models.list()
        print("\nAvailable models:")
        for model in models.data:
            if hasattr(model, 'id'):
                print(f"- {model.id}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    return client

def test_llama_processing(image_path: str, debug: bool = True, max_retries: int = 2):
    """Test Llama's form processing with detailed logging"""
    print(f"\n=== Testing Llama Processing ===")
    print(f"Image: {image_path}")
    
    for retry in range(max_retries + 1):
        if retry > 0:
            print(f"\nRetry attempt {retry}/{max_retries}")
            
        try:
            # Setup client
            print("Setting up Llama client...")
            client = setup_llama_client()
            
            # Read and encode image
            print("Reading image file...")
            with open(image_path, "rb") as f:
                image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine mime type
            ext = Path(image_path).suffix.lower()
            mime_type = "image/jpeg"  # Default
            if ext == ".png":
                mime_type = "image/png"
            elif ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".webp":
                mime_type = "image/webp"
            
            data_url = f"data:{mime_type};base64,{base64_image}"
            print(f"Image format: {mime_type}")
            
            # Prepare prompt
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
            
            print("\nMaking API request to Llama...")
            try:
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://example.com",
                        "X-Title": "UTEL Form Extractor Test",
                    },
                    model="gpt-4-vision-preview",  # Try GPT-4 Vision instead
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
                    timeout=120
                )
                
                print("\nAPI Response received:")
                print(f"Response object type: {type(completion)}")
                print(f"Response object attributes: {dir(completion)}")
                
                if hasattr(completion, 'model'):
                    print(f"Model used: {completion.model}")
                if hasattr(completion, 'usage'):
                    print(f"Usage stats: {completion.usage}")
                
            except Exception as api_error:
                print(f"\nAPI Request Error: {api_error}")
                if retry < max_retries:
                    print("Will retry...")
                    continue
                return None
            
            print("\nProcessing response...")
            if not completion:
                print("Error: No completion in response")
                if retry < max_retries:
                    continue
                return None
                
            if not completion.choices:
                print("Error: No choices in completion")
                print(f"Full completion object: {completion}")
                if retry < max_retries:
                    continue
                return None
                
            response_text = completion.choices[0].message.content
            if debug:
                print("\nRaw response:")
                print(response_text)
                
            # Try to parse JSON
            try:
                response_data = json.loads(response_text)
                print("\nParsed JSON successfully:")
                print(json.dumps(response_data, indent=2))
                
                # Convert to Pydantic model
                result = UTELFormData(**response_data)
                print("\nConverted to Pydantic model successfully")
                return result
                
            except json.JSONDecodeError as e:
                print(f"\nError parsing JSON: {e}")
                print("Raw response that failed JSON parsing:")
                print(response_text)
                if retry < max_retries:
                    continue
                return None
                
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
            if retry < max_retries:
                continue
            return None
    
    print("\nAll retry attempts failed")
    return None

if __name__ == "__main__":
    # Get image path from command line or use default
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use the first image in the images directory
        image_dir = Path("images")
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if not images:
            print("No images found in ./images directory")
            sys.exit(1)
        image_path = str(images[0])
    
    # Run test
    result = test_llama_processing(image_path)
    
    if result:
        print("\n=== Final Result ===")
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print("\n=== Test Failed ===")
        print("No valid result was returned")
