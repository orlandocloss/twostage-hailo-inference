import os
import cv2
import numpy as np
import json
from dotenv import load_dotenv
from sensing_garden_client import SensingGardenClient

# --- Main Function ---
def main():
    """
    Initializes the client and attempts to upload a single, hardcoded payload
    to debug the 400 Bad Request error.
    """
    print("--- Starting Upload Debug Script ---")

    # 1. Load Environment Variables
    load_dotenv()
    api_key = os.environ.get('SENSING_GARDEN_API_KEY')
    if not api_key:
        print("Error: SENSING_GARDEN_API_KEY not found in .env file.")
        return

    # 2. Initialize the Sensing Garden Client
    try:
        sgc = SensingGardenClient(
            base_url=os.environ.get('API_BASE_URL'),
            api_key=api_key,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.environ.get('AWS_REGION', 'us-east-1')
        )
        print("SensingGardenClient initialized successfully.")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # 3. Create a simple, valid dummy JPG image
    # This eliminates any potential issues with the live frame capture.
    dummy_image_array = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image_array[:, :, 2] = 255  # Make it blue
    _, buffer = cv2.imencode('.jpg', dummy_image_array)
    image_data = buffer.tobytes()
    print(f"Created a dummy JPG image of {len(image_data)} bytes.")

    # 4. Hardcode the exact payload that previously failed
    # Using the data from your error log.
    # Note: The timestamp has been set to a valid, recent time.
    payload = {
        'device_id': "test_new_pipeline",
        'model_id': "london_141",
        'image_data': image_data,
        'timestamp': "2024-07-29T10:00:00.000000", # Using a valid, non-future timestamp
        'family': "Vespidae",
        'genus': "Vespa",
        'species': "Vespula vulgaris",
        'family_confidence': 0.25333958864212036,
        'genus_confidence': 0.18767708539962769,
        'species_confidence': 0.3242592513561249,
        'bbox': [0.4546296296296296, 0.5476851851851852, 0.3111111111111111, 0.3138888888888889],
        'track_id': 1
    }
    print("\n--- Hardcoded Payload to be Sent ---")
    # Print a clean version of the payload for review
    debug_payload = payload.copy()
    debug_payload['image_data'] = f"<... {len(payload['image_data'])} bytes ...>"
    for key, value in debug_payload.items():
        print(f"  - {key}: {value} (type: {type(value).__name__})")
    print("------------------------------------")


    # 5. Attempt the upload
    try:
        print("\nAttempting to upload the hardcoded payload...")
        
        # This call mimics the logic from the working client_processing.py script
        sgc.classifications.add(
            device_id=payload['device_id'],
            model_id=payload['model_id'],
            image_data=payload['image_data'],
            family=payload.get("family"),
            genus=payload.get("genus"),
            species=payload.get("species"),
            family_confidence=payload.get("family_confidence"),
            genus_confidence=payload.get("genus_confidence"),
            species_confidence=payload.get("species_confidence"),
            timestamp=payload['timestamp'],
            bounding_box=payload.get('bbox'), # Use the list directly
            track_id=float(payload['track_id']) if payload.get('track_id') is not None else None
        )
        print("\n--- ✅ SUCCESS ---")
        print("The upload was successful. This indicates the issue is likely related to data processing in the live `run.py` script and not the API call itself.")

    except Exception as e:
        print("\n--- ❌ FAILURE ---")
        print(f"The upload failed with the same error: {e}")
        print("This suggests the problem is with the data format itself, even when hardcoded.")
        print("Next steps: Double-check the API documentation for the exact data types and required fields.")


if __name__ == "__main__":
    main()
