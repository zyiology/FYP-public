import requests
import re
import pandas as pd
import time
import os


def get_image_pkey(mapillary_url):
    match = re.search(r'pKey=([0-9]+)', mapillary_url)
    if match:
        return match.group(1)
    return


def retrieve_image_info(pkey, requests_per_minute=60):
    app_access_token = 'MLY|6940940835984902|5c4a505b2a240f74e2d0225ea5e4bf0d'#'MLY|24924485573816932|806a1a6757c095d5a7917b2ef0863788'
    query_url = f'https://graph.mapillary.com/{pkey}?fields=thumb_original_url,is_pano&access_token={app_access_token}'
    # query_url = f'https://graph.mapillary.com/{pkey}?fields=thumb_original_url'
    # headers = {"Authorization": f"OAuth {app_access_token}"}
    headers = {}
    response = requests.get(query_url, headers)
    data = response.json()
    time.sleep(60 / requests_per_minute)
    print(data)
    print(query_url)
    return (data['thumb_original_url'], data['is_pano'])


def retrieve_mapillary_urls(data_file, output_file, requests_per_minute):
    df = pd.read_csv(data_file, sep='\t', header=0)
    #df['image_filename'] = df['Country'] + '_' + df['City'] + '_' + df['Postal code']

    image_urls = []
    image_indices = []

    with open(output_file, 'w') as f:
        f.write("index" + '\t' + "is_pano" + '\t' + "url" + '\n')

        for i, mapillary_url in enumerate(df['Image link']):
            # skip if link is not defined
            if not mapillary_url or not mapillary_url.startswith("https://"):
                continue

            pkey = get_image_pkey(mapillary_url)
            if not pkey:
                print(f"pkey not defined for {mapillary_url}")
                continue

            image_url, is_pano = retrieve_image_info(pkey, requests_per_minute)
            image_urls.append(image_url)
            image_indices.append(i)

            f.write(str(i))
            f.write('\t')
            f.write(str(int(is_pano)))
            f.write('\t')
            f.write(image_url)
            f.write('\n')

    return image_indices, image_urls


# Function to download an image from a URL
def download_image(index, url, save_folder):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{save_folder}/image_{index}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error: {e}")


# Function to download a list of images from URLs with rate limiting
def download_images_from_list(indices_list, url_list, save_folder, requests_per_minute):
    for index, url in zip(indices_list, url_list):
        download_image(index, url, save_folder)
        time.sleep(60 / requests_per_minute)


# def find_ispano(data_file, links_file):
#     df = pd.read_csv(data_file, sep='\t', header=0)
#
#     links_df = pd.read_csv(links_file, sep='\t', header=0)
#
#     for i in links_df['index']:

def download_from_pkey(pkeys, ids):
    for pkey, id in zip(pkeys,ids):
        img_url, is_pano = retrieve_image_info(pkey)
        download_image(str(id), img_url, 'images/manual_dl/')



if __name__ == "__main__":
    req_per_minute = 60

    get_image_urls = False
    image_urls_file = 'image_links_v2.txt'

    if get_image_urls:

        indices, urls = retrieve_mapillary_urls('Building_data_sampling.tsv', image_urls_file, req_per_minute) #need to make it handle the file correctly - it's reading float into url if url is empty
        print(f"image urls extracted to '{image_urls_file}'")

    download_images = False
    if download_images:
        image_folder = "images/"
        os.makedirs(image_folder, exist_ok=True)

        df = pd.read_csv(image_urls_file, sep='\t', header=0)

        download_images_from_list(df['index'], df['url'], image_folder, req_per_minute)

    custom_curl = True
    if custom_curl:
        custom_pkeys = ['562256951799207']
        ids = [60]
        download_from_pkey(custom_pkeys, ids)




#https://www.ntu.edu.sg/?code=AQD9jOUyV2iRyuHJcNA-TBwXnU86ZCGIOpzISyYcBmHbWdXCCP9-bRcgGvcCuwL9qaImNpn35npWu6JaBem80TX8x3le18PpW26Cd0Bto6TNXNABltpKAhnUDTobltlPAVk7e5yTPJrHgArdlDWDdi58XO3U527t5GlVK2Imkc20YbE9qseRcFflkzr8BNipXdYnOGYfjAVH0-FjYziANb3uWR7bVeeIf2lHL_aX2aTZHwc7qKCKixauXB_0FrwvWXlCv5fhczkVhr_IORl_Hg3Uz0crQHL5t0GllNZbT3WGlA