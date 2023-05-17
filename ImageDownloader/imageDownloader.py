import requests
import os

# set up the API key and endpoint URL
api_key = "NzvQvhmhUHXqahgl82joGMyv0x8vDdChEhVrddp6vDMA7WQFqXKWEZxo"
url = "https://api.pexels.com/v1/search"

# set up the parameters for the API request
query = "Person"  # Tag
per_page = 80  # the Pexels API allows maximum 80
StartPage = 1  # Straing page (max 200 request/hours)
EndPage = 10  # Ending page (max 200 request/hours)
headers = {"Authorization": api_key} 

# specify the directory to save the images in
save_dir = "E:/OnlabKepek/"+query+"/"

# create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# loop through the range of pages to download
for i in range(StartPage, EndPage):
    # set up the parameters for the current page of results
    params = {"query": query, "per_page": per_page, "page": i}
    # send a request to the Pexels API to get the current page of results
    response = requests.get(url, headers=headers, params=params)
    # parse the JSON response to get the list of photos
    data = response.json()
    photos = data["photos"]
    # loop through the list of photos and download each one
    for j, photo in enumerate(photos):
        # create the URL for the current photo
        photo_url = photo["src"]["original"]
        # send a request to download the photo
        response = requests.get(photo_url)
        # save the photo to the specified directory
        with open(save_dir + f"{i*per_page+j+1}.jpg", "wb") as f:
            f.write(response.content)