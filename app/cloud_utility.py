from google.cloud import storage
from configparser import ConfigParser
import os
from pathlib import Path

config = ConfigParser()
config.read("config.ini")

storage_client = storage.Client.from_service_account_json(config.get("settings","credential_json_path"))

def download_from_gcs(bucket_name):
    if not os.path.exists("./models"):
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs()
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            Path(f"models/{directory}").mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(f"models/{blob.name}")

def get_blob_list(bucket_name):
    bucket = storage_client.bucket(bucket_name)
    print("Blobs:")
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob)
    print("Listed all storage buckets.")

if __name__ == "__main__":
    bucket_name = config.get("cloud_params", "bucket_name")
    source_blob_name = config.get("cloud_params", "faiss_model")
    # get_blob_list(bucket_name)
    download_from_gcs(bucket_name, source_blob_name)