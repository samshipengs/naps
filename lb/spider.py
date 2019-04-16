import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
from datetime import datetime
from selenium.webdriver.firefox.options import Options


def main():
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options, executable_path=r'./geckodriver')
    lb = 'https://recsys.trivago.cloud/leaderboard/'
    driver.get(lb)
    
    trs = driver.find_elements_by_class_name('tr')
    trs_parse = [[j.text for j in i.find_elements_by_class_name('td')]  for i in trs]
    records = pd.DataFrame(trs_parse, columns=['rank', 'team_name', 'score'])
    # hourly
    ct = datetime.now()#.strftime('%Y-%m-%d %H')
    records['timestamp'] = ct

    filename = 'records.csv'
    if os.path.isfile(filename):
        previous_records = pd.read_csv(filename)
        records = pd.concat([previous_records, records], axis=0, ignore_index=True)
    
    records.to_csv(filename, index=False) 
    print(records)




if __name__ == '__main__':
    main()
