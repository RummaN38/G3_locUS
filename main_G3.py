import os
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import time
from utils.G3 import G3

# CONFIG
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "checkpoints/g3.pth"

# SAVE_DIR = "downloaded_images"
# os.makedirs(SAVE_DIR, exist_ok=True)

# STABLE DATA (WIKIMEDIA)
DATA = [
    {
        "name": "asia_tajmahal",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/Taj-Mahal.jpg",
        "lat": 27.1751,
        "lon": 78.0421
    },
    {
        "name": "europe_eiffel",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg",
        "lat": 48.8584,
        "lon": 2.2945
    },
    {
        "name": "north_america_statue",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg",
        "lat": 40.6892,
        "lon": -74.0445
    },
    {
        "name": "south_america_christ",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Christ_the_Redeemer_-_Cristo_Redentor.jpg",
        "lat": -22.9519,
        "lon": -43.2105
    },
    {
        "name": "africa_pyramids",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Kheops-Pyramid.jpg",
        "lat": 29.9792,
        "lon": 31.1342
    },
    {
        "name": "australia_opera",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/7c/Sydney_Opera_House_-_Dec_2008.jpg",
        "lat": -33.8568,
        "lon": 151.2153
    },
    {
        "name": "antarctica",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/07/View_of_the_Riiser-Larsen_Ice_Shelf_in_Antarctica.jpg",
        "lat": -75.2500,
        "lon": 0.0000
    }
]

# DOWNLOAD IMAGE

# def download_and_save_image(url, filename):
#     headers = {"User-Agent": "Mozilla/5.0"}
#
#     response = requests.get(url, headers=headers, timeout=15)
#
#     if response.status_code != 200:
#         raise Exception(f"HTTP Error: {response.status_code} for {url}")
#
#     if "image" not in response.headers.get("Content-Type", ""):
#         raise Exception(f"Invalid content type for {url}")
#
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#
#     path = os.path.join(SAVE_DIR, filename)
#     img.save(path)
#
#     return img

def load_image_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        raise Exception(f"HTTP Error: {response.status_code} for {url}")

    if "image" not in response.headers.get("Content-Type", ""):
        raise Exception(f"Invalid content type for {url}")

    return Image.open(BytesIO(response.content)).convert("RGB")


# LOAD MODEL

def load_model():
    model = G3(device).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

# IMAGE EMBEDDING

def get_image_embedding(model, image):
    image_tensor = model.vision_processor(
        images=image,
        return_tensors='pt'
    )['pixel_values'].reshape(3, 224, 224)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        image_embeds = model.vision_projection_else_2(
            model.vision_projection(
                model.vision_model(image_tensor)[1]
            )
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    return image_embeds

# GPS GRID

def generate_world_grid(step=5):
    latitudes = np.arange(-90, 90, step)
    longitudes = np.arange(-180, 180, step)

    gps_list = []
    for lat in latitudes:
        for lon in longitudes:
            gps_list.append([lat, lon])

    return np.array(gps_list)

# GPS EMBEDDINGS

def get_location_embeddings(model, gps_candidates):
    gps_tensor = torch.tensor(gps_candidates).float().to(device)

    with torch.no_grad():
        location_embeds = model.location_encoder(gps_tensor)
        location_embeds = model.location_projection_else(location_embeds)
        location_embeds = location_embeds / location_embeds.norm(dim=-1, keepdim=True)

    return location_embeds

# HAVERSINE DISTANCE

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# CONTINENT MAPPING

def get_continent(lat, lon):
    if -170 <= lon <= -30:
        return "North America" if lat >= 15 else "South America"
    elif -30 < lon <= 60:
        if lat >= 35:
            return "Europe"
        else:
            return "Africa"
    elif 60 < lon <= 180:
        return "Asia" if lat >= 0 else "Australia"
    return "Antarctica"

# MAIN

def run():
    print("Loading model...")
    model = load_model()

    print("Building GPS grid...")
    gps_candidates = generate_world_grid(step=5)
    location_embeds = get_location_embeddings(model, gps_candidates)

    for item in DATA:
        print("\n==============================")
        print(f"Processing: {item['name']}")

        # image = download_and_save_image(
        #     item["url"],
        #     item["name"] + ".jpg"
        # )

        image = load_image_from_url(item["url"])

        time.sleep(1)  # prevent 429 (important)

        image_embeds = get_image_embedding(model, image)

        similarity = torch.matmul(image_embeds, location_embeds.T)
        similarity = similarity.squeeze(0).cpu().numpy()

        # TOP-K RESULTS

        top_k = 5
        top_indices = similarity.argsort()[-top_k:][::-1]

        print("\n--- TOP PREDICTIONS ---")
        for rank, idx in enumerate(top_indices):
            lat, lon = gps_candidates[idx]
            score = similarity[idx]
            error_km = haversine(item["lat"], item["lon"], lat, lon)

            print(f"{rank+1}. Lat: {lat:.2f}, Lon: {lon:.2f} | Score: {score:.4f} | Error: {error_km:.2f} km")

        # FINAL RESULT

        best_idx = top_indices[0]
        pred_lat, pred_lon = gps_candidates[best_idx]

        final_error = haversine(item["lat"], item["lon"], pred_lat, pred_lon)
        pred_continent = get_continent(pred_lat, pred_lon)

        print("\n--- FINAL RESULT ---")
        print(f"Predicted Location: ({pred_lat:.4f}, {pred_lon:.4f})")
        print(f"Predicted Continent: {pred_continent}")
        print(f"Ground Truth: ({item['lat']}, {item['lon']})")
        print(f"Final Error: {final_error:.2f} km")

# RUN

if __name__ == "__main__":
    run()
