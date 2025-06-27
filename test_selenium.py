from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

print("Attempting to launch Chrome...")
try:
    # Automatically download and manage ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    print("Chrome launched successfully. Opening google.com...")
    driver.get("https://www.google.com")
    time.sleep(5) # Keep browser open for 5 seconds

    print("Closing Chrome...")
    driver.quit()
    print("Selenium test finished successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure Google Chrome is installed and updated.")
    print("If you see an error about ChromeDriver, check your Chrome version.")