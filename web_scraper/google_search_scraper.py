from googlesearch import search
import re
import csv
from urllib.request import urlopen
import httplib2
from bs4 import BeautifulSoup, SoupStrainer

DOWNLOAD_DIRECTORY = "scraped_reports"

# query: query string that we want to search for.
# TLD: TLD stands for the top-level domain which means we want to search our results on google.com or google. in or some other domain.
# lang: lang stands for language.
# num: Number of results we want.
# start: The first result to retrieve.
# stop: The last result to retrieve. Use None to keep searching forever.
# pause: Lapse to wait between HTTP requests. Lapse too short may cause Google to block your IP. Keeping significant lapses will make your program slow but it’s a safe and better option.
# Return: Generator (iterator) that yields found URLs. If the stop parameter is None the iterator will loop forever.


# import pandas as pd
# table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# df = table[0]
# df.to_csv('S&P500-Info.csv')
# df.to_csv("S&P500-Names.csv", columns=['Security'])

# check if the url contains the domain of the company
def validate_url_domain(company_name, url):
	pass

def is_pdf_url(url):
	return re.search(".pdf$", url)

# check if the url is really a sustainability report pdf
def validate_pdf_url:
	pass

def download_file(company_name, url):
	filename = f"{company_name}_report"
	response = urlopen(url)
	file = open(f"{DOWNLOAD_DIRECTORY}/{filename}.pdf", 'wb')
	file.write(response.read())
	file.close()
	print(f"downloaded {filename}")

# return whether a file has been downloaded
def open_website(url):
	has_download = False
	http = httplib2.Http()
	status, response = http.request(url) # may need to set a timeout here, and check status

	for link in BeautifulSoup(response, 'html.parser', parse_only=SoupStrainer('a')):
		if link.has_attr('href'):
			pdf_url = link['href']
			if is_pdf_url(pdf_url) and validate_pdf_url(pdf_url): # could be more than 1 pdf, need extra smart validations, e.g. check if the file contains words like "sustainability", "report"
				download_file(company_name, pdf_url)
				has_download = True
	return has_download

def main():
	filename = 'S&P500-Names.csv'
	s_and_p_500_list = []

	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for (index, name) in reader:
			s_and_p_500_list.append(name) # add name of the company

	for company_name in s_and_p_500_list[:10]: # first 10 companies
		query = f"{company_name} sustainability report"
		print(f"{query}:")
		for url in search(query, num=3, stop=3, pause=2): # get the first 3 url
		    print(f"\t{url}")
		    if is_pdf_url(url):
		    	download_file(company_name, url)
		    	break # go to next company
		    else:
		    	# go inside the website and find pdf
		    	if open_website(url): # if downloaded a file
		    		break

if __name__ == "__main__":
	main()



