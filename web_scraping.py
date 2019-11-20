from selenium import webdriver
from selenium.webdriver.common.by import by
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected conditions as expected as EC 
from selenium.common.exceptions import TimeoutException

# https://medium.com/the-andela-way/introduction-to-web-scraping-using-selenium-7ec377a8cf72

"""
1st import: Allows you to launch/initialise a browser.
2nd import: Allows you to search for things using specific parameters.
3rd import: Allows you to wait for a page to load.
4th import: Specify what you are looking for on a specific page in order to determine that the webpage has loaded.
5th import: Handling a timeout situation                  """

option = webdriver.ChromeOptions()
option.add_argument(“ — incognito”)


browser = webdriver.Chrome(executable_path=’/Library/Application Support/Google/chromedriver’,
						   chrome_options=option)

browser.get("https://github.com/TheDancerCodes")
timeout = 15

try:
    # Wait until the final element [Avatar link] is loaded.
    # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
    # the last things to be loaded.
    WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//img[@class='avatar width-full rounded-2']")))
except TimeoutException:
    print("Timed out waiting for page to load")
    browser.quit()


