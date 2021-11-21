import time
import re
from selenium import webdriver
from bs4 import BeautifulSoup
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def init_driver():
    # Initiate the driver
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.wait = WebDriverWait(driver, 10)
    return driver

def send_request(url, driver):
    # Send organized requests to Gutenberg.org so that books of a certain category
    # can be obtained.
    queue = []
    types = ['Classics', 'Tragedy', 'Science Fiction', 'Fantasy', 'Fairytale', 'Adventure', 
            'Crime & Mystery', 'Historical Fiction', 'Humor', 'Fictional Diaries', 'Satire', 
            'Romance', 'Horror', 'Dystopian', 'Biography', 'Memoirs', 'Analysis', 'Philosophy',
            'Psycology', 'Economics', 'Reference']
    driver.get(url)
    
    for t in types:
        try:
            # Send the keyword in the query section 
            query = driver.wait.until(EC.presence_of_element_located(
            (By.NAME, 'query')))
            button = driver.wait.until(EC.element_to_be_clickable(
            (By.ID, 'search-button')))
            query.clear()
            query.send_keys(t)
            button.click()

            # Add the link of the book into the queue
            for _ in range(2):
                links = driver.wait.until(EC.presence_of_all_elements_located(
                (By.CLASS_NAME, 'link')))
                for link in links:
                    if link not in queue and re.match('http://www.gutenberg.org/ebooks/(\d)+$', \
                    link.get_attribute('href')):
                        queue.append((t, link.get_attribute('href')))
                # Click the 'next page' button
                nextPageButton = driver.wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="content"]/div[2]/div/ul/li[32]/div/span/a')))
                time.sleep(2)
                nextPageButton.click()
            print('Link acquired for' + t)
        except TimeoutException:
            pass

    return queue

def get_books(queue, driver):
    # Get the content of the books whose urls have been saved in the queue
    while len(queue) > 0:
        type_ = queue[0][0]
        driver.get(queue[0][1])
        del queue[0]
        
        # Get the url for the plain text of the book 
        texts = driver.wait.until(EC.presence_of_all_elements_located(
        (By.CLASS_NAME, 'link')))
        for text in texts:
            if not re.match('\w+[.]txt[.]utf-8$',text.get_attribute('href')):
                del text

        # Get the title of the book
        title = driver.wait.until(EC.presence_of_element_located(
        (By.CLASS_NAME, 'header'))).text

        # Write the content into a text file
        with open('./gutenberg/{}:{}.txt'.format(type_, title), 'w+', encoding="utf-8") as f:
            url = texts[0].get_attribute('href')
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, features="html.parser")
            f.write(soup.get_text().strip())
            f.close()
            print(type_, title)

            
def main():
    driver = init_driver()
    # Acquire the books at gutenberg.org
    queue = send_request('http://www.gutenberg.org/ebooks/', driver)
    get_books(queue, driver)

    driver.close()

main()