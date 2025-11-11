import os
import pandas as pd
from PIL import Image
import google.generativeai as genai

# ========== STEP 1: Gemini API Key ==========
genai.configure(api_key="AIzaSyDNq1sts47AGQazTqFPKq2AfMXNY90tAT4")  

# ========== STEP 2: Configuration ==========
image_folder = r"C:\Users\kisho\Desktop\SEM 4\CS496\UGP2\Images_Kishore\Riddles\rem2"     
output_csv = "kishore_riddles.csv"
model_name = "gemini-1.5-flash"
prompt = "This is a brain teaser or puzzle. Please solve it and explain your reasoning step-by-step."

# ========== STEP 3: Load Model ==========
model = genai.GenerativeModel(model_name)

# ========== STEP 4: Load Previous Data if Exists ==========
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv, sep='|')
    processed_images = set(existing_df['image_id'])
else:
    existing_df = pd.DataFrame(columns=["image_id", "reasoning", "inference"])
    processed_images = set()

# ========== STEP 5: Process New Images ==========
results = []
image_files = sorted([f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for img_name in image_files:
    if img_name in processed_images:
        print(f"Skipping already processed image: {img_name}")
        continue

    image_path = os.path.join(image_folder, img_name)

    try:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        reasoning = response.text.strip()
    except Exception as e:
        reasoning = f"Error: {e}"

    results.append({
        "image_id": img_name,
        "reasoning": reasoning,
        "inference": ""  # Empty column for now
    })

# ========== STEP 6: Append New Results ==========
if results:
    new_df = pd.DataFrame(results)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(output_csv, index=False, sep='|')
    print(f"\n Appended {len(new_df)} new responses to '{output_csv}'.")
else:
    print("\n No new images to process.")
