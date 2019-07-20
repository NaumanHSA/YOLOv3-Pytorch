from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"weapon,knife,pistol","limit":100,"chromedriver":"C:\\Users\\nauma\\OneDrive\\Desktop\\yolov3workflow-master\\1_WebImage_Scraping\\chromedriver.exe","print_urls":True}   #creating list of arguments
#Make sure to set chromedrive is the appropriate director if you want to download more than 100 images
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
